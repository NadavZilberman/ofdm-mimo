import matplotlib.pyplot as plt
import numpy as np
import random
import helpers as hp

# FOR EACH STA: qam symbols -> add preamble -> zero pad -> ifft -> add cp -> dod -> channel -> doa

# add in rx -> remove cp -> fft -> mimo detection separately

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

    # ---------------------- STA0 Tx ---------------------- #

    # create qam stream for sta0
    sta0_payload = np.random.choice(symbols_list, (NUM_SYMBOLS, NUM_SC_STA0))
    sta0_preamble = np.random.choice(symbols_list, (1, NUM_SC_STA0))
    sta0_symbols_mat = np.concatenate((sta0_preamble, sta0_payload))

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
    channel_sta0_to_rx = np.array([create_random_channel(10, MAX_CHAN_DS) for i in range(N_RX*NUM_STREAMS_STA0)])
    
    # convolve with channel and add DOA
    doa_sta0_array = get_uni_random_angle(4)
    signal_in_rx = np.zeros((N_RX, len(time_signal_flattened)+channel_sta0_to_rx.shape[1]-1), dtype=complex)
    for i in range(N_RX):
        channel_i = channel_sta0_to_rx[i,:]
        signal_in_rx[i,:] = np.convolve(time_signal_flattened, channel_i) * doa_sta0_array[i]

    # back to matrix shape
    matrix_of_symbols = np.zeros(((N_RX,)+ sym_to_transmit_with_cp.shape), dtype=complex)
    for i in range(N_RX):
        matrix_of_symbols[i,:] = np.reshape(signal_in_rx[i,:-channel_sta0_to_rx.shape[1]+1], sym_to_transmit_with_cp.shape)
    
    # get rid of CP
    matrix_of_symbols_no_cp = matrix_of_symbols[:,:, L_CP:]
    
    # channel estimation
    estimated_qam_symbols_with_channel = np.fft.fftshift(np.fft.fft(matrix_of_symbols_no_cp, n=N_FFT, axis=2), axes=2)[:,:,:50] # do fft, and take only relevant SCs
    h_est_sta0 = estimated_qam_symbols_with_channel[:,0,:]/sta0_preamble[:,]

    # estimate symbols (running over sub-carriers)
    s_hat = np.zeros_like(sta0_symbols_mat)
    for sc in range(NUM_SC_STA0):
        s_hat[:, sc] = np.sum(np.conj(h_est_sta0[:,sc][:,np.newaxis]) * estimated_qam_symbols_with_channel[:, :, sc], axis=0)/np.linalg.norm((h_est_sta0[:,sc]))**2

    # EVM calculation for STA0
    evm = np.mean(np.abs(s_hat - sta0_symbols_mat)**2)
    print(f"STA0: EVM = {evm}")

if __name__ == "__main__":
    main()