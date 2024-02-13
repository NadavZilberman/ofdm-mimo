import matplotlib.pyplot as plt
import numpy as np
import helpers as hlp

def main():
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5
    snr_array_dB = np.arange(0, 41)
    snr_array = 10**(snr_array_dB/10)
    REP = 5000000
    h_const = 1

    # rho_array = 1/snr_array**0.5

    SER_awgn = []
    SER_rayleigh = []


    random_symbols = np.random.choice(symbols_list, REP)
    s = random_symbols

    for i, snr in enumerate(snr_array):
        rho = 1/np.sqrt(snr)
        n = (np.random.standard_normal(REP) + 1j*np.random.standard_normal(REP))/2**0.5
        h_rayleigh = (np.random.standard_normal(REP) + 1j*np.random.standard_normal(REP))/2**0.5

        # Detection: with AWGN #
        y = h_const * s + rho * n
        s_hat = y/h_const
        s_estimation = hlp.estimate_closest_symbols(s_hat, symbols_list)

        ser = hlp.calculate_SER(s_estimation, s, REP)
        SER_awgn.append(ser)

        # Detection: with AWGN & Rayleigh channel #
        y = h_rayleigh * s + rho * n
        s_hat = y/h_rayleigh
        s_estimation = hlp.estimate_closest_symbols(s_hat, symbols_list)

        ser = hlp.calculate_SER(s_estimation, s, REP)
        SER_rayleigh.append(ser)



    plt.figure()
    plt.plot(snr_array_dB, SER_awgn, label="SISO: AWGN")
    plt.plot(snr_array_dB, SER_rayleigh, label="SISO: AWGN and Rayleigh Channel")
    plt.grid(which='both', axis='both')
    plt.legend()
    plt.xticks(snr_array_dB)
    plt.yscale('log')
    plt.ylabel('SER')
    plt.xlabel('SNR[dB]')
    plt.title(f"SER vs SNR - SISO\n{REP} repetitions")
    plt.show()
    pass

if __name__ == "__main__":
    main()