import matplotlib.pyplot as plt
import numpy as np
import helpers as hp

def random_qam_vector(size, order):
    max_value = np.sqrt(order) - 1
    values = np.arange(-max_value, max_value + 1, 2)
    real_values = np.random.choice(values, size)
    imag_values = np.random.choice(values, size)
    output = real_values + 1j*imag_values
    return output

def create_ccdf(papr, papr_array):
    ccdf = np.zeros_like(papr_array)
    for i, curr_papr in enumerate(papr_array):
        ccdf[i] = np.sum(papr>curr_papr)
    return ccdf/len(papr)

def main():
    N_symbols = 100000

    # --- Fixing oversampling and modulation, changing n_fft --- #
    plt.figure()
    os = 4
    order=64
    N_fft = [64, 256, 1024]
    M_allocations = [60, 200, 1000]
    papr_array_dB_to_compare = np.arange(5, 13, 0.01)
    for i, n_fft in enumerate(N_fft):
        stream = random_qam_vector((N_symbols, M_allocations[i]), order)
        stream_oversampled = np.zeros((N_symbols, M_allocations[i]*os), dtype=complex)
        stream_oversampled[:,:int(M_allocations[i]/2)] = stream[:,:int(M_allocations[i]/2)]
        stream_oversampled[:,-int(n_fft/2):] = stream[:,-int(n_fft/2):]
        stream_shifted = np.fft.ifftshift(stream_oversampled, axes=1)
        time_signal = np.fft.ifft(stream_shifted, axis=1)
        power = np.abs(time_signal)**2
        PAPR = np.max(power, axis=1)/np.average(power, axis=1)
        PAPR_dB = 10*np.log10(PAPR)
        ccdf = create_ccdf(PAPR_dB, papr_array_dB_to_compare)
        plt.plot(papr_array_dB_to_compare, ccdf, label=f"M={M_allocations[i]}, N_FFT={n_fft}, {order}QAM (OS={os})")
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.yscale('log')
    plt.title(f'PAPR CCDF: OFDM, {order}QAM, with M allocated tones\n{N_symbols} OFDM symbols')
    plt.ylabel('CCDF')
    plt.xlabel('PAPR0[dB]')
    plt.show()

    # --- Fixing M allocation tones, oversampling and N_FFT, changing modulation order --- #
    plt.figure()
    os = 4
    M = 200
    n_fft = 256
    modulation_order = [4, 16, 256]
    papr_array_dB_to_compare = np.arange(5, 11, 0.01)
    for order in modulation_order:
        stream = random_qam_vector((N_symbols, M), order)
        stream_oversampled = np.zeros((N_symbols, M*os), dtype=complex)
        stream_oversampled[:,:int(M/2)] = stream[:,:int(M/2)]
        stream_oversampled[:,-int(n_fft/2):] = stream[:,-int(n_fft/2):]
        stream_shifted = np.fft.ifftshift(stream_oversampled, axes=1)
        time_signal = np.fft.ifft(stream_shifted, axis=1)
        power = np.abs(time_signal)**2
        PAPR = np.max(power, axis=1)/np.average(power, axis=1)
        PAPR_dB = 10*np.log10(PAPR)
        ccdf = create_ccdf(PAPR_dB, papr_array_dB_to_compare)
        plt.plot(papr_array_dB_to_compare, ccdf, label=f"M={M}, N_FFT={n_fft}, {order}QAM (OS={os})")
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.yscale('log')
    plt.title(f'PAPR CCDF: OFDM, with M={M} allocated tones\nDifferent modulation orders\n{N_symbols} OFDM symbols')
    plt.ylabel('P(PAPR > PAPR0)')
    plt.xlabel('PAPR0[dB]')
    plt.show()

   # --- Fixing M allocation tones, N_FFT and Modulation order, changing oversampling rate --- #
    plt.figure()
    order = 16
    M = 100
    n_fft = 128
    oversampling = [1, 2, 4, 8, 16]
    papr_array_dB_to_compare = np.arange(7, 13, 0.01)

    for os in oversampling:
        stream = random_qam_vector((N_symbols, M), order)
        stream_oversampled = np.zeros((N_symbols, M*os), dtype=complex)
        stream_oversampled[:,:int(M/2)] = stream[:,:int(M/2)]
        stream_oversampled[:,-int(n_fft/2):] = stream[:,-int(n_fft/2):]
        stream_shifted = np.fft.ifftshift(stream_oversampled, axes=1)
        time_signal = np.fft.ifft(stream_shifted, axis=1)
        power = np.abs(time_signal)**2
        PAPR = np.max(power, axis=1)/np.average(power, axis=1)
        PAPR_dB = 10*np.log10(PAPR)
        ccdf = create_ccdf(PAPR_dB, papr_array_dB_to_compare)
        plt.plot(papr_array_dB_to_compare, ccdf, label=f"M={M}, N_FFT={n_fft}, {order}QAM (OS={os})")
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.yscale('log')
    plt.title(f'PAPR CCDF: OFDM, {order}QAM with M={M} allocated tones, different oversampling rates\n{N_symbols} OFDM symbols')
    plt.ylabel('P(PAPR > PAPR0)')
    plt.xlabel('PAPR0[dB]')
    plt.show()

if __name__ == "__main__":
    main()

    # Questions:
    # 1. If we take less symbols than n_fft, the padding is in relation of n_fft or to num of symbols?