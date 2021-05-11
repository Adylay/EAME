import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import gmtime, strftime
from scipy.signal import butter, lfilter

def filter_bandpass(EKG_data, threshold_down, threshold_up, f_signal, filter_order):
    threshold_down_nyquist = threshold_down / (0.5 * f_signal)
    threshold_up_nyquist = threshold_up / (0.5 * f_signal)
    b, a = butter(filter_order, [threshold_down_nyquist, threshold_up_nyquist], btype="band", output='ba')
    y = lfilter(b, a, EKG_data)
    return y

def peak_detection(EKG, margin=1, limit=None):
    EKG_length = EKG.size
    auxiliary_vector = np.zeros(EKG_length + 2 * margin)
    auxiliary_vector[:margin] = EKG[0] - 1.e-6
    auxiliary_vector[-margin:] = EKG[-1] - 1.e-6
    auxiliary_vector[margin:margin + EKG_length] = EKG
    introductory_point = np.zeros(EKG_length)
    introductory_point[:] = True

    for s in range(margin):
        start = margin - s - 1
        h_b = auxiliary_vector[start: start + EKG_length]  # before
        start = margin
        h_c = auxiliary_vector[start: start + EKG_length]  # central
        start = margin + s + 1
        h_a = auxiliary_vector[start: start + EKG_length]  # after
        introductory_point = np.logical_and(introductory_point, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(introductory_point)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[EKG[ind] > limit]
    return ind

def plot_data(axis, data, title='', fontsize=10):
        axis.set_title(title, fontsize=fontsize)
        axis.grid(which='both', axis='both', linestyle='--')
        axis.plot(data, color="black", zorder=1)


def plot_points(axis, values, indices):
        axis.scatter(x=indices, y=values[indices], c="red", s=10, zorder=2)