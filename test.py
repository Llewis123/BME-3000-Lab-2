#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class code for BME300, class 9.
Makes a sine wave and plays it through the speakers.

Created on Tue Sep 26 10:26:20 2023

@author: djangraw
"""

# import
from matplotlib import pyplot as plt
import numpy as np
import sounddevice as sd

# create a time array
fs = 44100 # sampling frequency in Hz
dt = 1/fs
t = np.arange(start=-5, stop = 5, step = dt)
# declare sine wave frequency
f = 261.6256 # middle C


# define a function to create a sine array
def get_sine(t,f,phi=0):
    """
    Returns a building-block sine signal.

    Parameters
    ----------
    t : array of floats of size sample_count
        the times of each sample in seconds.
    f : float
        frequency of desired sine wave in Hz.
    phi : float (optional)
        phase of desired sine wave in radians. The default is 0.

    Returns
    -------
    sine : array of floats of size sample_count
        the signal value at each sample.

    """
    sine = np.sin(2*np.pi * f * t + phi)
    return sine

# get sine signal
my_sine = get_sine(t,f)

# plot result
plt.figure(33,clear=True)
plt.plot(t,my_sine)

# add zero lines
plt.plot([-6,6],[0,0],'k') # horizontal line
plt.plot([0,0],[-1.25,1.25],'k') # vertical line


# annotate plot
plt.xlabel('time (s)')
plt.ylabel('signal (A.U.)')
plt.grid()
plt.title('sine function')
plt.tight_layout()

# play sine wave
sd.play(my_sine,samplerate=fs)
