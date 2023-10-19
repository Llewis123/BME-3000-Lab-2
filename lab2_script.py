# -*- coding: utf-8 -*-

"""
Test script that creates and plots a convolution against a time array,
whilst showing the 2 building signals, allows us to
run a  drug simulation of cascade sytem and plot the results,
and lastly uses arrays and a sound package to allow us to
manipulate and convolve sounds together.
@author: Nick Taylor and Lincoln Lewisquerre

"""

# %% Import Packages

from matplotlib import pyplot as plt

import numpy as np

import soundfile as sf

import sounddevice as sd

import lab2_module as l2m

# %% Part 1: Convolution Properties: Plotting

dt = 0.01

time = np.arange(0, 5.01, dt)

# Create Input Signal (Sine Wave)

input_signal = np.sin(6 * np.pi * time)

plt.plot(input_signal)

# Create System impulse

system_impulse = l2m.rect_pulse(time, width=1.5, centered=False, start=0.5, end=2)

plt.plot(system_impulse)

# Scale Input Signal

input_signal_scaled = 2 * input_signal

plt.plot(input_signal_scaled)
plt.show()
# Convolve Signals

fig, axs = plt.subplots(3, 3, figsize=(12, 12))

fig.suptitle("Visualization of Convolution Properties")

# Plots in Row 0

axs[0, 0].plot(time, input_signal, label='Input Signal')

axs[0, 0].set_title('Input Signal')

axs[0, 1].plot(time, system_impulse, label='System Impulse')

axs[0, 1].set_title("System Impulse")

convolution_result1 = np.convolve(input_signal, system_impulse, mode='full')

axs[0, 2].plot(l2m.get_time_convolution(convolution_result1, dt), convolution_result1, label='Convolution')

axs[0, 2].set_title('Convolution (Input * Impulse)')

# Plots in Row 1

axs[1, 0].plot(time, system_impulse, label='System Impulse')

axs[1, 0].set_title("System Impulse")

axs[1, 1].plot(time, input_signal, label='Input Signal')

axs[1, 1].set_title('Input Signal')

convolution_result2 = np.convolve(system_impulse, input_signal, mode='full')

axs[1, 2].plot(l2m.get_time_convolution(convolution_result2, dt), convolution_result2, label='Convolution')

axs[1, 2].set_title('Convolution (Impulse * Input)')

# Plots in Row 2

axs[2, 0].plot(time, input_signal_scaled, label='Scaled Input Signal ')

axs[2, 0].set_title("Scaled Input Signal")

axs[2, 1].plot(time, system_impulse, label='System Impulse')

axs[2, 1].set_title('System Impulse')

convolution_result3 = np.convolve(input_signal_scaled, system_impulse, mode='full')

axs[2, 2].plot(l2m.get_time_convolution(convolution_result3, dt), convolution_result3, label='Convolution')

axs[2, 2].set_title('Convolution (Scaled Input * Impulse)')

# Add x and y Labels

for ax in axs.flat:
    ax.set_xlabel('Time (s)')

    ax.set_ylabel('Amplitude (A.U.)')

# Organize Graphs

plt.tight_layout()

# Save Figure

plt.savefig('pictures/input_impulse_covolition.png')
plt.show()

# %% Part 2: Build a Convolution Function

# first use module to get the convolved signal
# reminder that we use our optional variable, which we could set as nump=0 if the order
# of the variables in the function was not obvious.
my_convolved_signal = l2m.get_convolved_signal(input_signal, system_impulse, 0)
# then just simply plot
plt.figure(3)
plt.plot(l2m.get_time_convolution(my_convolved_signal, dt), my_convolved_signal)
plt.title('Convolution (Input * Impulse)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (A.U.)')
plt.show()
# %% Part 3: Simplify a Cascade System

# declare variables
dt = 0.01
drug_time = np.arange(0, 50 + dt, dt)

# calculate drug dosage equation
drug_dosage = 1 - np.cos(0.25 * np.pi * drug_time)

# create the arrays of each system transfer function
h1 = np.array(0.25 * (np.exp(-drug_time / 4)) * drug_dosage)
h2 = np.array(1 - np.cos(0.25 * np.pi * -drug_time))
h3 = np.array(np.exp((-2 * (drug_time - 1) ** 2)))

drug_convolution1 = np.convolve(h1, h2)

body_impulse = np.convolve(drug_convolution1, h3)

plt.figure(4)
plt.title("Drug Concentration Response")

# Testing Amplitudes
test_amp = [0, 1, 2, 3]

# Testing Denominators
test_denom = [2, 4, 6]

# for loop to iterate through denom and amplitudues to print and run
# the drug simulation to them
for denom in test_denom:
    for amps in test_amp:
        x_t = amps - np.cos(1 / denom * np.pi * drug_time)
        label = f'denominator = {denom}, amplitude = {amps}'
        l2m.run_drug_simulations(x_t, body_impulse, dt, label)

# plot them
plt.xlabel('Time (s)')
plt.ylabel('Drug Concentration')
plt.legend()
plt.show()
plt.tight_layout()
plt.savefig('pictures/drug_concentration.pdf')

# %%
'''
Part 4: Play own sound


I commented out te plays except for the last one so that I did not have
to hear a nightmare everytime I run this block of code
'''
# load in the data
file_path_omg = 'data/omg_sound.wav'
# assign our data and sample rate variables by using sf to read the files
data_omg, sample_rate_omg = sf.read(file_path_omg)
"""
# I multiplied the data by 2 because it was very quiet before
sd.play(data_omg*2, sample_rate_omg)
sd.wait()
"""

# This will be our time array
time_omg = np.arange(0, len(data_omg)) * (1 / sample_rate_omg)
# taking the mean as it is a 2D array (2 channels of audio)
mean_data_omg = np.mean(data_omg, axis=1)

# plot the results and save the figure
plt.plot(time_omg, mean_data_omg)
plt.title("Mean OMG data vs time")
plt.xlabel("Time in seconds")
plt.ylabel("Amplitude A.U.")
plt.grid(visible=True)
plt.autoscale(enable=True)
plt.savefig("pictures/Mean_OMG_Data_vs_Time.png")
plt.show()
print(
    f"\n This graph looks a lot like the sound signature of the noise would be. It seems just like random sinusoidal "
    f"waves, and at a high frequency.")

# now play with a doubled sampling rate
print(f"With a doubled sample rate, their will be less time, so signal will be compressed and much faster")
"""
sd.play(data_omg*2, sample_rate_omg*2)             
sd.wait()
"""
print(f"\n That is exactly what happened!")
# the audio did not last as long and ended up shortening it


# now play with a halved sampling rate
print(
    f"\n With a halved sampling rate, the time will be scaled upwards by 2, increasing the length of signal and "
    f"stretching it")
"""
sd.play(data_omg*2, sample_rate_omg*0.5)
sd.wait()
"""
print(f"\n This is exactly what happened!")

# insert print here
# %%

"""
Here we load in the high pass filter and convolve it with our data

"""
# load in the high pass filter
file_path_hpf = "filters/HPF_1000Hz_fs44100_n10001.txt"
hpf = np.loadtxt(file_path_hpf)

# we added an extra optional to the function so that we could just use the np.convolve in there instead of double
# for loop, so it takes a lot less time. We got Adam's approval on this.
omg_convolve_hpf = l2m.get_convolved_signal(hpf, mean_data_omg, 1)

# save the convolution
np.savetxt("data/highpassed.txt", omg_convolve_hpf)

sd.play(omg_convolve_hpf * 10000000000, sample_rate_omg)
sd.wait()
print(f"This filter got rid of all lower frequencies, and essentially made the higher frequency noise louder")
# %%
"""
Here we now do the low pass filter
"""
# load in low pass filter
file_path_lpf = "filters/LPF_1000Hz_fs44100_n10001.txt"
lpf = np.loadtxt(file_path_lpf)

# now convolve the highpass filter with the lowpass filter
omg_convolve_lpf = l2m.get_convolved_signal(lpf, mean_data_omg, 1)

# save the convolution
np.savetxt("data/lowpassed.txt", omg_convolve_lpf)

sd.play(omg_convolve_lpf * 1000000000, sample_rate_omg)
sd.wait()

# insert print statement here
print(f"This filter got rid of all of the higher frequencies and kept the lower frequencies. It seems \n\
that it made lower frequency noise louder by eliminating the higher frequency noise")

# %%
"""
Here we will get the unknown filter and convolve it 
"""
# lettuce create a vector that goes from 0 to 0.02 back down to 0
# we use concatenate to combine two arrays, one going up to 0.02 and the other going down to 0
upper_lim = np.arange(0, 0.02, 0.0004)
lower_lim = np.arange(0.02, 0, -0.0004)
strange_filter = np.concatenate((upper_lim, lower_lim))
omg_convolve_strange = l2m.get_convolved_signal(strange_filter, mean_data_omg, 1)
# save the convolution
np.savetxt("data/bandpassed.txt", omg_convolve_strange)

sd.play(omg_convolve_strange * 10000000, sample_rate_omg)
sd.wait()
print(f"This is a bandpass filter from 85-255Hz, which essentially attenuates any frequency outside of \n\
those values. It seems like this filter is adding noise, which perhaps it because normally, the noise cancels out, which is maybe guassian distr??? \n\
Anyways, it added some noise into the system.")
# add print here

# %%
"""
Here will we create our h array of zeros except for two indexes and then convolve it
"""
# lettuce create the array h
h = np.zeros(10002)
h[0] = 1
h[-1] = 1
# we get the convolved signal
omg_convolve_h = l2m.get_convolved_signal(mean_data_omg, h, 1)
# save the convolution
np.savetxt("data/hed.txt", omg_convolve_h)
# now I am multipying it by a lot just so I can hear it
sd.play(omg_convolve_h * 10000000, sample_rate_omg)
sd.wait()
# add print here
print(f"This completely nulled my response and essentially turned it off, in non-technical terms, \n\
It kind of sounds like the short sound you get when you turn on a speaker, like the speakers transient voltage response to being turned on")

# %%
''' 
This section we will load my cat talking and convolve it with the oh my god sound
'''

file_path_meow = "data/penelope_talking.wav"
data_meow, sample_rate_meow = sf.read(file_path_meow)
"""
sd.play(data_meow, sample_rate_meow)
sd.wait()
"""

# In order to convolve, we will convolve the data meow from 0.25 seconds in to 0.25 +
# the duration of the omg data to capture my cats meow
data_meow_mean = np.mean(data_meow, axis=1)
# in samples, this would be 0.25 * sampling rate in order to get number of samples
# before we cut it off
# and then for the end point, it is just the size of the omg data + the cut off point
num_samples_cutoff = int(0.25 * sample_rate_meow)
num_samples_cutoff_end = len(data_omg) + num_samples_cutoff
data_meow_chopped = data_meow_mean[num_samples_cutoff:num_samples_cutoff_end]

# now we can convolve the two
omg_convolve_meow = l2m.get_convolved_signal(data_meow_chopped, mean_data_omg, 1)

# save the convolution
np.savetxt("data/meowed.txt", omg_convolve_meow)

# play the sound!!
sd.play(data_omg * 10000000, sample_rate_meow)
sd.wait()
# insert print statement here
