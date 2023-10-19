"""

Module that allows us to plot a rectangular pulse,
make a convoluted signal using 2 input signals and make the time array for that convolution,
as well as running a drug simulation using a cascade system.
@author: Nick Taylor and Lincoln Lewisquerre

"""

import numpy as np
from matplotlib import pyplot as plt


def rect_pulse(t, width=1, centered=True, amplitude=1, start=0, end=1):
    """
    Parameters
    ----------
    t : array, size = (501,)
        Array of time points at which the pulse is plotted.
    width : float, optional
        Tells the function how wide the rectangular pulse needs to be. The default is 1.
    centered : boolean, optional
        Dictates if the pulse is centered at (0,0) or other on the graph. The default is True.
    amplitude : float, optional
        Gives the height of the rectangular pulse. The default is 1.
    start : float , optional
        The desired point at which the pulse will start. The default is 0.
    end : float, optional
        The desired point at which the pulse will end. The default is 1.



    Returns
    -------
    array
        returns the rectangular pulse at the desired start and end points with the correct amplitude, width, and centeredness.

    """

    # essentially, if we want it centered around the x axis @ 0, we do it this way.
    if centered:
        # np.where is really useful for manipulating not only rect pulse but
        # all the other basic signals you could need
        return np.where((t >= start - width / 2) & (t <= end + width / 2), amplitude, 0)
    else:
        # if not centered you must specify where it starts at
        return np.where((t >= start) & (t <= start + width), amplitude, 0)


def get_convolved_signal(input_signal, system_impulse, nump):
    """
    Parameters
    ----------
    input_signal : array, size = (501,)
        Array of values of the input signal.
    system_impulse : array, size = (501,)
        Array of values of the system impulse.



    Returns
    -------
    my_convolved_signal : array, size (1002,)
        The given array created by the convolution of the 2 signals used.



    """

    # create an array of empty numbers that represents teh input_signal + system_impulse lengths
    my_convolved_signal = np.zeros(len(input_signal) + len(system_impulse) - 1)
    # this is for our optional parameter, just useful so we do not repeat any code.
    if nump == 0:
        # double for loop as instructed to iterate through the input_signal and system_impulse
        for element_signal_index in range(len(input_signal)):
            for element_impulse_index in range(len(system_impulse)):
                # This is our implementation of the convolution formula
                # we add each index to get the index of the convolved signal and assign that value to be the input
                # signal element times the impulse element
                my_convolved_signal[element_signal_index + element_impulse_index] += input_signal[
                                                                                         element_signal_index] * \
                                                                                     system_impulse[
                                                                                         element_impulse_index]
    elif nump == 1:
        # this is for if we just want to use np.convolve
        my_convolved_signal = np.convolve(np.mean(input_signal), system_impulse, mode="full")
    return my_convolved_signal


def get_time_convolution(convolution_result, dt):
    """
    Parameters
    ----------
    convolution_result : array, size = (501,)
        The array of the convolution which can be used to graph.
    dt : float
        The "step" of which the time increases by.



    Returns
    -------
    time_convolution : array, size = (501,)
        The array of times created to properly graph the convolutions.



    """

    time_convolution = np.arange(0, len(convolution_result)) * dt
    return time_convolution


def run_drug_simulations(input_signal, system_impulse, dt, label):
    '''
    Parameters
    ----------
    input_signal : array, size = (501,)
        Array of values of the input signal.
    system_impulse : array, size (501,)
        Array of values of the system impulse.
    dt : float
        The "step" of which the time increases by.
    label : string
        Allows for the legend to properly label the different lines on our graph.



    Returns
    -------
    None.



    '''
    drug_convolution = np.convolve(input_signal, system_impulse, mode='full')
    time = np.arange(0, len(drug_convolution)) * dt
    plt.plot(time, drug_convolution, label=label)
    plt.legend()
# %%
