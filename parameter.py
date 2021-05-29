filter_params = {
    'signal_frequency': 255,
    'filter_lowcut': 5,
    'filter_highcut': 15.0,
    'filter_order': 2
}
convolve_params = {
    'integration_window': 15
}

detection_params = {
    'findpeaks_limit_processed': 0.35,
    'findpeaks_limit_filter': 0.5,
    'findpeaks_spacing': 50
}
qrs_params ={
    'refractory_period': 100,
    'threshold_value': 30
}