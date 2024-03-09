import matplotlib.pyplot as plt
import numpy as np
import helpers as hlp
from numpy import linalg

def main():
    plt.figure()
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5
    snr_array_dB = np.arange(0, 26)
    snr_array = 10**(snr_array_dB/10)
    REP = 5000000
    N_ARRAY = [2, 4]
    for rx_array_len in N_ARRAY:
        SER_array = []
        random_symbols = np.random.choice(symbols_list, (REP, 1))

        for snr in snr_array:
            rho = 1/np.sqrt(snr)
            h = (np.random.standard_normal((REP, rx_array_len)) + 1j*np.random.standard_normal((REP, rx_array_len)))/2**0.5
            n = (np.random.standard_normal((REP, rx_array_len)) + 1j*np.random.standard_normal((REP, rx_array_len)))/2**0.5
            y = h * random_symbols + rho * n
            # calc s_hat
            s_hat = np.sum(np.conj(h[:,:,np.newaxis]) * y[:,:,np.newaxis], axis=(1, 2))/np.linalg.norm(h, axis=1)**2
            # # estimate symbols
            s_estimation = hlp.estimate_closest_symbols(s_hat, symbols_list)
            # calculate SER
            curr_SER = hlp.calculate_SER(s_estimation, random_symbols.T, REP)
            SER_array.append(curr_SER)

        plt.plot(snr_array_dB, SER_array, label=f"SER: N={rx_array_len}")

    plt.grid(which='both', axis='both')
    plt.legend()
    plt.xticks(snr_array_dB)
    plt.yscale('log')
    plt.title(f'SER vs SNR - MRC with N=2,4 Antennas\n{REP} repetitions')
    plt.ylabel('SER')
    plt.xlabel('SNR[dB]')
    plt.show()
    pass

if __name__ == "__main__":
    main()
