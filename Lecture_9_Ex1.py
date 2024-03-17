import matplotlib.pyplot as plt
import numpy as np
import helpers as hp


def main():
    N_FFT = 256
    L_CP = 16
    N = 20
    SNR = 20
    # fft_linspace = np.linspace(int(-fft_size/2), int(fft_size/2), 1000)
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5

    # stream creation
    pilot = np.random.choice(symbols_list, (1, N_FFT))
    payload = np.random.choice(symbols_list, (N, N_FFT))
    stream = np.concatenate((pilot, payload))

    stream_shifted = np.fft.fftshift(stream, axes=1)
    # channels creation
    channel_0 = np.zeros((L_CP, 1), dtype=complex)
    channel_0[0] = 1
    channel_1 = np.zeros((L_CP, 1), dtype=complex)
    channel_1[0] = 1
    channel_1[9] = 0.9j
    channels = [channel_0, channel_1]

    # stream coding
    time_signal = np.fft.ifft(stream_shifted, n=N_FFT, axis=1)
    time_signal_with_cp = np.concatenate((time_signal[:,-L_CP:], time_signal), axis=1)
    time_signal_flattened = time_signal_with_cp.flatten()
    # apply channel
    for channel in channels:
        signal_in_rx = np.convolve(time_signal_flattened, channel[:,0])

        # noise
        rho = 1/np.sqrt(SNR)
        noise = (np.random.standard_normal(signal_in_rx.shape)) + 1j*np.random.standard_normal(signal_in_rx.shape)/2**0.5 # after signal exists
        signal_in_rx = signal_in_rx #+ rho*noise

        # back to matrix shape
        matrix_of_symbols = np.reshape(signal_in_rx[:-len(channel)+1], time_signal_with_cp.shape)
        
        # channel estimation
        estimated_qam_symbols_with_channel = np.fft.fftshift(np.fft.fft(matrix_of_symbols, n=N_FFT, axis=1), axes=1)
        

        # estimated_qam_symbols_with_channel_no_cp = estimated_qam_symbols_with_channel[:, L_CP:]
        
        h_est = np.fft.fft(estimated_qam_symbols_with_channel[0])/np.fft.fft(pilot.T[:,0])

        # TODO DOENS'T LOOK LIKE THE REAL CHANNEL
        estimated_ifft_coeffs = estimated_qam_symbols_with_channel[1:,:]/h_est
        estimated_ifft_coeffs_flattened = estimated_ifft_coeffs.flatten()
        payload_flatten = payload.flatten()
        payload_estimation = hp.estimate_closest_symbols(estimated_ifft_coeffs_flattened, symbols_list)
        evm = np.mean(np.abs(payload_estimation - payload_flatten))
if __name__ == "__main__":
    main()