import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from function import filter_bandpass, peak_detection, plot_data, plot_points
from parameter import filter_params, convolve_params, detection_params

ECG_path = "ecg_data/ecg_data_1.csv"
ECG_data = pd.read_csv(ECG_path, skiprows=0)
ECG_raw = ECG_data['ecg_measurement']

ECG_filter = filter_bandpass(ECG_raw,
                             threshold_down=filter_params['filter_lowcut'],
                             threshold_up=filter_params['filter_highcut'],
                             f_signal=filter_params['signal_frequency'],
                             filter_order=filter_params['filter_order'])

ECG_diff = np.ediff1d(ECG_filter)
ECG_squared = ECG_diff ** 2
ECG_convolve = np.convolve(ECG_squared, np.ones(convolve_params['integration_window']))

# _____________________________________DETEKCJA_PIKÓW________________________ #

ECG_peaks_index = peak_detection(EKG=ECG_convolve,
                                 limit=detection_params['findpeaks_limit'],
                                 margin=detection_params['findpeaks_spacing'])

ECG_peaks_values = ECG_convolve[ECG_peaks_index]


fig, axarr = plt.subplots(6, sharex=True, figsize=(15, 18))

plot_data(axis=axarr[0], data=ECG_raw, title='ECG_raw')

plot_data(axis=axarr[1], data=ECG_filter, title='ECG_filter')

plot_data(axis=axarr[2], data=ECG_diff, title='ECG_diff')

plot_data(axis=axarr[3], data=ECG_squared, title='ECG_squared')

plot_data(axis=axarr[4], data=ECG_convolve, title='ECG_convolve')
plot_points(axis=axarr[4], values=ECG_convolve, indices=ECG_peaks_index)
plt.show()