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

def peak_detection(EKG_processed, EKG_filter, margin=1, limit_processed=None, limit_filter=None):
    EKG_processed_length = EKG_processed.size
    EKG_filter_length = EKG_filter.size

    auxiliary_vector_EKG_processed = np.zeros(EKG_processed_length + 2 * margin)
    auxiliary_vector_EKG_processed[:margin] = EKG_processed[0] - 1.e-6
    auxiliary_vector_EKG_processed[-margin:] = EKG_processed[-1] - 1.e-6
    auxiliary_vector_EKG_processed[margin:margin + EKG_processed_length] = EKG_processed
    introductory_point_processed = np.zeros(EKG_processed_length)
    introductory_point_processed[:] = True

    auxiliary_vector_EKG_filter = np.zeros(EKG_filter_length + 2 * margin)
    auxiliary_vector_EKG_filter[:margin] = EKG_filter[0] - 1.e-6
    auxiliary_vector_EKG_filter[-margin:] = EKG_filter[-1] - 1.e-6
    auxiliary_vector_EKG_filter[margin:margin + EKG_filter_length] = EKG_filter
    introductory_point_filter = np.zeros(EKG_filter_length)
    introductory_point_filter[:] = True

    for s in range(margin):
        start = margin - s - 1
        h_b = auxiliary_vector_EKG_processed[start: start + EKG_processed_length]  # before
        start = margin
        h_c = auxiliary_vector_EKG_processed[start: start + EKG_processed_length]  # central
        start = margin + s + 1
        h_a = auxiliary_vector_EKG_processed[start: start + EKG_processed_length]  # after
        introductory_point_processed = np.logical_and(introductory_point_processed, np.logical_and(h_c > h_b, h_c > h_a))

    for s in range(margin):
        start = margin - s - 1
        h_b = auxiliary_vector_EKG_filter[start: start + EKG_filter_length]  # before
        start = margin
        h_c = auxiliary_vector_EKG_filter[start: start + EKG_filter_length]  # central
        start = margin + s + 1
        h_a = auxiliary_vector_EKG_filter[start: start + EKG_filter_length]  # after
        introductory_point_filter = np.logical_and(introductory_point_filter, np.logical_and(h_c > h_b, h_c > h_a))

    index_processed = np.argwhere(introductory_point_processed)
    index_processed = index_processed.reshape(index_processed.size)

    index_filter = np.argwhere(introductory_point_filter)
    index_filter = index_filter.reshape(index_filter.size)

    if limit_processed is not None:
        index_processed = index_processed[EKG_processed[index_processed] > limit_processed]


    if limit_filter is not None:
        index_filter = index_filter[EKG_filter[index_filter] > limit_filter]

    k = 0
    index = np.copy(index_processed)
    index[index >= 0] = 0

    for i in range(len(index_processed)):
        for j in range(len(index_filter)):
            if abs(index_processed[i]-index_filter[j]) < 30:
                index[k] = index_processed[i]
                k = k+1

    index = index[:k]
    return index

def plot_data(axis, data, title='', fontsize=10):
        axis.set_title(title, fontsize=fontsize)
        axis.grid(which='both', axis='both', linestyle='--')
        axis.plot(data, color="black", zorder=1)


def plot_points(axis, values, indices):
        axis.scatter(x=indices, y=values[indices], c="red", s=10, zorder=2)

def detect_qrs(ECG_peaks_index, ECG_peaks_values, refractory_period, threshold_value):
    QRS_peaks_index_LAST = 0
    i = 1
    QRS_index = np.zeros(len(ECG_peaks_index)*len(ECG_peaks_values))
    QRS_values = QRS_index
    for x, y in zip(ECG_peaks_index, ECG_peaks_values):

        if x - QRS_peaks_index_LAST > refractory_period or QRS_peaks_index_LAST == 0:
            if y > threshold_value:
                QRS_index[i] = x
                QRS_values[i] = y
            else:
                QRS_index[i] = x
                QRS_values[i] = QRS_values[i-1]

        QRS_peaks_index_LAST = QRS_index[i]
        i = +1
    return QRS_index
