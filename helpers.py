import numpy as np

def estimate_closest_symbols(s_hat, symbols_list):
    closest_indices = np.abs(symbols_list - s_hat[:, np.newaxis]).argmin(axis=1)
    s_estimation = symbols_list[closest_indices]
    return s_estimation

def calculate_SER(s_estimation, s, num_of_rep):
    num_of_errors = np.sum(np.abs(s_estimation - s) > 0)
    return num_of_errors/num_of_rep