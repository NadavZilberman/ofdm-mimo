import matplotlib.pyplot as plt
import numpy as np
import helpers as hp


def main():
    N_FFT = 256
    L_CP = 16
    N = 20
    SNR = 10**(20/10)
    REP = 1000
    # fft_linspace = np.linspace(int(-fft_size/2), int(fft_size/2), 1000)
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5
    #
    evm_ch_0 = 0
    evm_ch_1 = 0
    evm_ch_0_perfect = 0
    evm_ch_1_perfect = 0
    # stream creation

    for rep in range(REP):
        pilot = np.random.choice(symbols_list, (1, N_FFT))
        payload = np.random.choice(symbols_list, (N, N_FFT))
        stream = np.concatenate((pilot, payload))

        # channels creation
        channel_0 = np.zeros((L_CP, 1), dtype=complex)
        channel_0[0] = 1
        channel_1 = np.zeros((L_CP, 1), dtype=complex)
        channel_1[0] = 1
        channel_1[9] = 0.9j
        channels = [channel_0, channel_1]

        # stream coding
        stream_shifted = np.fft.ifftshift(stream, axes=1)
        time_signal = np.fft.ifft(stream_shifted, n=N_FFT, axis=1)
        time_signal_with_cp = np.concatenate((time_signal[:,-L_CP:], time_signal), axis=1)
        time_signal_flattened = time_signal_with_cp.flatten()
        # apply channel
        for channel in channels:
            signal_in_rx = np.convolve(time_signal_flattened, channel[:,0])
            signal_power=np.average(np.abs(signal_in_rx)**2)

            # noise
            rho = signal_power/SNR
            noise = (np.random.standard_normal(signal_in_rx.shape)) + 1j*np.random.standard_normal(signal_in_rx.shape)/2**0.5 # after signal exists
            signal_in_rx = signal_in_rx + rho*noise

            # back to matrix shape
            matrix_of_symbols = np.reshape(signal_in_rx[:-len(channel)+1], time_signal_with_cp.shape)
            
            matrix_of_symbols_no_cp = matrix_of_symbols[:, L_CP:] # NOTE: we get rid of CP before calculating the symbols. It only serves us in channel convolution.
            
            # channel estimation - non perect
            estimated_qam_symbols_with_channel = np.fft.fftshift(np.fft.fft(matrix_of_symbols_no_cp, n=N_FFT, axis=1), axes=1)

            h_est = estimated_qam_symbols_with_channel[0]/pilot.T[:,0]
            
            # symbols estimation - non perfect
            estimated_ifft_coeffs = estimated_qam_symbols_with_channel[1:,:]/h_est
            estimated_ifft_coeffs_flattened = estimated_ifft_coeffs.flatten()
            payload_flatten = payload.flatten()
            payload_estimation = hp.estimate_closest_symbols(estimated_ifft_coeffs_flattened, symbols_list)

            # calculate evm
            evm = np.mean(np.abs(payload_estimation - payload_flatten)**2)

            # channel estimation - perfect

            h_est = np.fft.fftshift(np.fft.fft(channel, n=N_FFT, axis=0)).T # NOTE: fftshift here is important
            estimated_ifft_coeffs = estimated_qam_symbols_with_channel[1:,:]/h_est

            # symbols estimation - perfect
            estimated_ifft_coeffs_flattened = estimated_ifft_coeffs.flatten()
            payload_flatten = payload.flatten()
            payload_estimation = hp.estimate_closest_symbols(estimated_ifft_coeffs_flattened, symbols_list)

            # calculate evm
            evm_perfect_ch_knowledge = np.mean(np.abs(payload_estimation - payload_flatten)**2)

            if channel is channel_0:
                evm_ch_0 += evm/REP
                evm_ch_0_perfect += evm_perfect_ch_knowledge

            if channel is channel_1:
                evm_ch_1 += evm/REP
                evm_ch_1_perfect += evm_perfect_ch_knowledge

    print("evm ch 0 = ", evm_ch_0)
    print("evm ch 1 = ", evm_ch_1)
    print("evm ch 0 with perfect channel knowledge = ", evm_ch_0_perfect)
    print("evm ch 1 with perfect channel knowledge = ", evm_ch_1_perfect)
if __name__ == "__main__":
    main()