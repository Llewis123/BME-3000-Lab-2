"""

Module that allows us to plot a rectangular pulse,
make a convoluted signal using 2 input signals and make the time array for that convolution,
as well as running a drug simulation using a cascade system.
@author: Nick Taylor and Lincoln Lewisquerre

"""


import numpy as np
from matplotlib import pyplot as plt


def rect_pulse(t, width=1, centered=True, amplitude=1, start=0, end=1):
    if centered:
        return np.where((t >= start - width / 2) & (t <= end + width / 2), amplitude, 0)
    else:
        return np.where((t >= start) & (t <= start + width), amplitude, 0)


def get_convolved_signal(input_signal, system_impulse, nump):
    my_convolved_signal = np.zeros(len(input_signal) + len(system_impulse) - 1)
    if nump == 0:
        index_convolved = 0
        for element_signal_index in range(len(input_signal)):
            for element_impulse_index in range(len(system_impulse)):
                # my_convolved_signal[index_convolved] += input_signal[element_signal_index] * system_impulse[
                #     index_convolved - element_signal_index]
                my_convolved_signal[element_signal_index + element_impulse_index] += input_signal[
                                                                                         element_signal_index] * \
                                                                                     system_impulse[
                                                                                         element_impulse_index]

        index_convolved += 1
    elif nump == 1:
        my_convolved_signal = np.convolve(np.mean(input_signal), system_impulse, mode="full")
    return my_convolved_signal

def get_time_convolution(convolution_result, dt):
    time_convolution = np.arange(0, len(convolution_result)) * dt
    return time_convolution


def run_drug_simulations(input_signal, system_impulse, dt, label):
    drug_convolution = np.convolve(input_signal, system_impulse)
    time = np.arange(0, len(drug_convolution), dt)
    plt.plot()
    plt.legend()
# %%
