import matplotlib.pyplot as plt
import numpy as np
import helpers as hlp

def main():
    plt.figure()
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5
    snr_array_dB = np.arange(0, 21)
    snr_array = 10**(snr_array_dB/10)
    REP = 1000000
    SIR_dB = 5
    pg = 10**(SIR_dB/10)/10 # given that signal_power_dB = 0
    [M, N] = [1, 4]
    
    SER_array_MRC = []
    SER_array_MVDR = []
    SER_array_MRC3_no_interference = []
    
    # MRC (copied and modified from Lecture2-Ex1.py)
    for snr in snr_array:
        rho = 1/np.sqrt(snr)
        random_s = np.random.choice(symbols_list, (REP, M, 1))
        noise = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5
        h = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5
        g = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5
        r = (np.random.standard_normal((REP, 1, 1)) + 1j*np.random.standard_normal((REP, 1, 1)))/2**0.5
        y = h * random_s + pg**0.5*g*r + rho * noise
        # calc s_hat
        s_hat = np.sum(np.conj(h[:,:,np.newaxis]) * y[:,:,np.newaxis], axis=(1, 2))/np.linalg.norm(h, axis=1)**2
        s_hat = s_hat[:,0]
        # # estimate symbols
        s_estimation = hlp.estimate_closest_symbols(s_hat, symbols_list)
        # calculate SER
        curr_SER = hlp.calculate_SER(s_estimation, random_s[:,0,0], REP)
        SER_array_MRC.append(curr_SER)
    plt.plot(snr_array_dB, SER_array_MRC, label="MRC")

    # MVDR
    for snr in snr_array:
        rho = 1/np.sqrt(snr)
        random_s = np.random.choice(symbols_list, (REP, M, 1))
        noise = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5
        h = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5
        g = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5
        r = (np.random.standard_normal((REP, 1, 1)) + 1j*np.random.standard_normal((REP, 1, 1)))/2**0.5
        y = h * random_s + pg**0.5*g*r + rho * noise

        I_mat = np.repeat(np.identity(n=N)[np.newaxis, :, :], repeats=REP, axis=0)
        C = pg * g * np.transpose(np.conj(g), (0, 2, 1)) + I_mat*rho**2
        
        # calculate s_hat
        h_conj = np.transpose(np.conj(h), (0, 2, 1))
        h_conj_C_inv = np.matmul(h_conj, np.linalg.inv(C))
        s_hat = np.matmul(h_conj_C_inv/np.matmul(h_conj_C_inv, h), y)
        s_hat = np.squeeze(s_hat, axis=(1,2))

        # estimate symbols
        s_estimation = hlp.estimate_closest_symbols(s_hat, symbols_list)

        # calculate SER
        curr_SER = hlp.calculate_SER(s_estimation, random_s[:, 0, 0], REP)
        SER_array_MVDR.append(curr_SER)
    plt.plot(snr_array_dB, SER_array_MVDR, label="MVDR")

    # MRC 3 without interference (copied and modified from Lecture2-Ex1.py)
    [M, N] = [1, 3]
    for snr in snr_array:
        rho = 1/np.sqrt(snr)
        random_s = np.random.choice(symbols_list, (REP, M, 1))
        noise = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5
        h = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5
        g = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5
        r = (np.random.standard_normal((REP, 1, 1)) + 1j*np.random.standard_normal((REP, 1, 1)))/2**0.5
        y = h * random_s + rho * noise

        # calc s_hat
        s_hat = np.sum(np.conj(h[:,:,np.newaxis]) * y[:,:,np.newaxis], axis=(1, 2))/np.linalg.norm(h, axis=1)**2
        s_hat = s_hat[:,0]

        # # estimate symbols
        s_estimation = hlp.estimate_closest_symbols(s_hat, symbols_list)
        
        # calculate SER
        curr_SER = hlp.calculate_SER(s_estimation, random_s[:,0,0], REP)
        SER_array_MRC3_no_interference.append(curr_SER)
    plt.plot(snr_array_dB, SER_array_MRC3_no_interference, label="MRC3 w/o interference")

    plt.grid(which='both', axis='both')
    plt.legend()
    plt.xticks(snr_array_dB)
    plt.yscale('log')
    plt.title(f'SER vs SNR: 4 RX Antennas, with interference Pg={SIR_dB}dB: MRC and MVDR\nCompared to MRC3 with no interference\n{REP} repetitions')
    plt.ylabel('SER')
    plt.xlabel('SNR[dB]')
    plt.show()
    pass

if __name__ == "__main__":
    main()
    breakpoint()