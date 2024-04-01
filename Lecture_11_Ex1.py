import matplotlib.pyplot as plt
import numpy as np
import random
import helpers as hp

# Tx: for each STA: qam symbols -> add preamble -> zero pad -> ifft -> add cp -> dod -> channel -> doa

# Rx: remove cp -> fft -> take relevant SCs -> estimate channel -> MIMO detection separately

def get_uni_random_angle(size=None):
    return np.random.uniform(0, 2*np.pi, size=size)

def create_random_channel(num_of_paths, max_delay):
    channel = np.zeros((max_delay), dtype=complex)
    phase_array = get_uni_random_angle(max_delay)
    random_delays = random.sample(range(max_delay), num_of_paths)
    for i, path_idx in enumerate(random_delays):
        channel[path_idx] = np.exp(-1j*2*np.pi*phase_array[i])
    return channel
    
def main():
    # OFDM config
    N_FFT = 256
    L_CP = 16
    N_RX = 4
    QAM_SIZE = 4 # QPSK
    NUM_SYMBOLS = 200
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5
    
    # Channel config
    MAX_CHAN_DS = L_CP

    # STAs config
    NUM_STREAMS_STA0 = 1
    NUM_SC_STA0 = 50
    NUM_STREAMS_STA1 = 2
    NUM_SC_STA1 = 20

    # ------------------------------------------------ STA0 Tx ------------------------------------------------ #

    # create qam stream for sta0
    sta0_payload = np.random.choice(symbols_list, (NUM_SYMBOLS, NUM_SC_STA0))
    sta0_preamble = np.random.choice(symbols_list, (1, NUM_SC_STA0))
    sta0_symbols_mat = np.concatenate((sta0_preamble, sta0_payload))

    # zero padding for the unused symbol by STA0
    sta0_zero_pad = np.zeros((NUM_SYMBOLS + 1, N_FFT-NUM_SC_STA0))
    sta0_stream_mat = np.concatenate((sta0_symbols_mat, sta0_zero_pad), axis=1)
    
    # stream coding
    stream_shifted = np.fft.ifftshift(sta0_stream_mat, axes=1)
    sym_to_transmit = np.fft.ifft(stream_shifted, n=N_FFT, axis=1)
    
    # add cp
    sym_to_transmit_with_cp = np.concatenate((sym_to_transmit[:,-L_CP:], sym_to_transmit), axis=1)
    time_signal_flattened = sym_to_transmit_with_cp.flatten()
    
    # add DOD
    dod_sta0 = get_uni_random_angle()
    time_signal_flattened = time_signal_flattened*np.exp(-1j*np.pi*np.sin(dod_sta0))

    # create multipath channels
    channel_sta0_to_rx = np.array([create_random_channel(10, MAX_CHAN_DS) for _ in range(N_RX)])
    
    # convolve with channel and add DOA
    doa_sta0_array = get_uni_random_angle(4)
    signal_in_rx = np.zeros((N_RX, len(time_signal_flattened)+channel_sta0_to_rx.shape[1]-1), dtype=complex)
    for i in range(N_RX):
        channel_i = channel_sta0_to_rx[i,:]
        signal_in_rx[i,:] = np.convolve(time_signal_flattened, channel_i) * doa_sta0_array[i]

    # ------------------------------------------------ STA1 Tx ------------------------------------------------ #

    # create qam streams for sta1
    sta1_payload_0 = np.random.choice(symbols_list, (NUM_SYMBOLS-1, NUM_SC_STA1)) # NUM_SYMBOLS-1 because we have 2 payloads and we want the streams to have the same length
    sta1_payload_1 = np.random.choice(symbols_list, (NUM_SYMBOLS-1, NUM_SC_STA1))
    sta1_preamble_0 = np.random.choice(symbols_list, (1, NUM_SC_STA1))
    sta1_preamble_1 = np.random.choice(symbols_list, (1, NUM_SC_STA1))
    sta1_symbols_mat_0 = np.concatenate((sta1_preamble_0, np.zeros((1,NUM_SC_STA1)), sta1_payload_0))
    sta1_symbols_mat_1 = np.concatenate((np.zeros((1,NUM_SC_STA1)), sta1_preamble_1, sta1_payload_1))

    # zero padding for the unused symbol by STA1
    sta1_zero_pad_before = np.zeros((NUM_SYMBOLS + 1, NUM_SC_STA0))
    sta1_zero_pad_after = np.zeros((NUM_SYMBOLS + 1, N_FFT-(NUM_SC_STA0+NUM_SC_STA1)))
    sta1_stream_mat_0 = np.concatenate((sta1_zero_pad_before, sta1_symbols_mat_0, sta1_zero_pad_after), axis=1)
    sta1_stream_mat_1 = np.concatenate((sta1_zero_pad_before, sta1_symbols_mat_1, sta1_zero_pad_after), axis=1)

    # streams coding
    stream_0_shifted = np.fft.ifftshift(sta1_stream_mat_0, axes=1)
    sym_to_transmit_0 = np.fft.ifft(stream_0_shifted, n=N_FFT, axis=1)
    stream_1_shifted = np.fft.ifftshift(sta1_stream_mat_1, axes=1)
    sym_to_transmit_1 = np.fft.ifft(stream_1_shifted, n=N_FFT, axis=1)

    # add cp
    sym_to_transmit_with_cp_0 = np.concatenate((sym_to_transmit_0[:,-L_CP:], sym_to_transmit_0), axis=1)
    time_signal_flattened_0 = sym_to_transmit_with_cp_0.flatten()
    sym_to_transmit_with_cp_1 = np.concatenate((sym_to_transmit_1[:,-L_CP:], sym_to_transmit_1), axis=1)
    time_signal_flattened_1 = sym_to_transmit_with_cp_1.flatten()

    # add DOD
    dod_sta1_0 = get_uni_random_angle()
    time_signal_flattened_0 = time_signal_flattened_0*np.exp(-1j*np.pi*np.sin(dod_sta1_0))
    dod_sta1_1 = get_uni_random_angle()
    time_signal_flattened_1 = time_signal_flattened_1*np.exp(-1j*np.pi*np.sin(dod_sta1_1))

    # create multipath channels
    channel_sta1_to_rx_0 = np.array([create_random_channel(10, MAX_CHAN_DS) for _ in range(N_RX)])
    channel_sta1_to_rx_1 = np.array([create_random_channel(10, MAX_CHAN_DS) for _ in range(N_RX)])

    # convolve with channel and add DOA
    doa_sta1_array_0 = get_uni_random_angle(4)
    doa_sta1_array_1 = get_uni_random_angle(4)
    # signal_in_rx = np.zeros((N_RX, len(time_signal_flattened_0)+channel_sta1_to_rx_0.shape[1]-1), dtype=complex)
    for i in range(N_RX):
        channel_i_0 = channel_sta1_to_rx_0[i,:]
        channel_i_1 = channel_sta1_to_rx_1[i,:]
        signal_in_rx[i,:] += np.convolve(time_signal_flattened_0, channel_i_0) * doa_sta1_array_0[i]
        signal_in_rx[i,:] += np.convolve(time_signal_flattened_1, channel_i_1) * doa_sta1_array_1[i]

    # ------------------------------------------------ Rx ------------------------------------------------ #
    # this part is mutual for all STAs
    
    # back to matrix shape
    matrix_of_symbols = np.zeros(((N_RX,)+ sym_to_transmit_with_cp.shape), dtype=complex)
    for i in range(N_RX):
        matrix_of_symbols[i,:] = np.reshape(signal_in_rx[i,:-channel_sta0_to_rx.shape[1]+1], sym_to_transmit_with_cp.shape)
    
    # get rid of CP
    matrix_of_symbols_no_cp = matrix_of_symbols[:,:, L_CP:]
    
    # demodulate
    estimated_qam_symbols_with_channel = np.fft.fftshift(np.fft.fft(matrix_of_symbols_no_cp, n=N_FFT, axis=2), axes=2)

    # ------------- STA 0 Channel Estimation and LS Symbols Estimation ------------- #

    # extract relevant SCs
    estimated_qam_symbols_with_channel_sta0 = estimated_qam_symbols_with_channel[:,:,:50]

    # estimate channel
    h_est_sta0 = estimated_qam_symbols_with_channel_sta0[:,0,:]/sta0_preamble[:,]

    # estimate symbols (running over sub-carriers)
    s_hat_sta0 = np.zeros_like(sta0_symbols_mat, dtype=complex)
    for sc in range(NUM_SC_STA0):
        s_hat_sta0[:, sc] = np.sum(np.conj(h_est_sta0[:,sc][:,np.newaxis]) * estimated_qam_symbols_with_channel_sta0[:, :, sc], axis=0)/np.linalg.norm((h_est_sta0[:,sc]))**2


    # ------------- STA 1 Channel Estimation and LS Symbols Estimation ------------- #
        
    # extract relevant SCs
    estimated_qam_symbols_with_channel_sta1 = estimated_qam_symbols_with_channel[:,:,50:70]

    # estimate channel
    h_est_sta0_0 = estimated_qam_symbols_with_channel_sta1[:,0,:]/sta1_preamble_0[:,]
    h_est_sta0_1 = estimated_qam_symbols_with_channel_sta1[:,1,:]/sta1_preamble_1[:,]
    
    s_hat_sta1 = np.zeros(((2,)+ sta1_symbols_mat_0.shape), dtype=complex)
    for sc in range(NUM_SC_STA1):
        H_curr_sc = np.vstack((h_est_sta0_0[:,sc],h_est_sta0_1[:,sc])).T
        H_conj = np.conj(H_curr_sc.T)
        s_hat_sta1[:, :, sc] = np.matmul(np.matmul(np.linalg.inv(np.matmul(H_conj, H_curr_sc)), H_conj), estimated_qam_symbols_with_channel_sta1[:,:,sc])

    # EVM calculation for STA0
    evm = np.mean(np.abs(s_hat_sta0 - sta0_symbols_mat)**2)
    print(f"STA0: EVM = {evm}")
    
    # EVM calculation for STA1
    sta1_symbols_both_streams = np.concatenate((sta1_symbols_mat_0[np.newaxis], sta1_symbols_mat_1[np.newaxis]))
    evm = np.mean(np.abs(s_hat_sta1 - sta1_symbols_both_streams)**2)
    print(f"STA1: EVM = {evm}")
    plt.figure()
    plt.xlabel("real")
    plt.ylabel("imag")
    plt.title("Constellation plot for STA0 estimated symbols")
    plt.scatter(np.real(s_hat_sta0.flatten()), np.imag(s_hat_sta0.flatten()))

    plt.figure()
    plt.xlabel("real")
    plt.ylabel("imag")
    plt.title("Constellation plot for STA1 estimated symbols")
    plt.scatter(np.real(sta1_symbols_both_streams.flatten()), np.imag(sta1_symbols_both_streams.flatten()))
    plt.show()
    # We can see that the EVM is practically zero and all
if __name__ == "__main__":
    main()