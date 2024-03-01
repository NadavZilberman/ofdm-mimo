import matplotlib.pyplot as plt
import numpy as np
import helpers as hlp
from Lecture_2_Ex2 import create_H

def main():
    plt.figure()
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5
    snr_array_dB = np.arange(0, 26)
    snr_array = 10**(snr_array_dB/10)
    REP = 500000
    ant_pairs = [[2,4], [2,2]] # [M, N]
    SER_array_SM = [[], [], []] # Reminder: make sure there's the same number of sub arrays
    SER_array_STC = []
    
    # SM with ML

    for i, [M, N] in enumerate(ant_pairs):
        if M == 2:
            combinations = np.array([[x, y] for x in symbols_list for y in symbols_list]).reshape(-1, M, 1)
        elif M == 4:
            combinations = np.array([[x, y, z, w] for x in symbols_list for y in symbols_list for z in symbols_list for w in symbols_list]).reshape(-1, M, 1)
        combinations_repeated = np.tile(combinations, (REP, 1, 1, 1))
        num_of_combinations = len(combinations)
        for snr in snr_array:
            rho = 1/np.sqrt(snr)

            # generate symbols
            random_s = np.random.choice(symbols_list, (REP, M, 1))

            # generate channels and noise
            H = (np.random.standard_normal((REP, N, M)) + 1j*np.random.standard_normal((REP, N, M)))/2**0.5
            noise = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5

            # calculate H_tild and repeat it
            H_tild = H/M**0.5
            H_tild_repeated = np.repeat(H_tild[:, np.newaxis], repeats=num_of_combinations, axis=1)

            # calculate y and repeat it
            y = np.matmul(H_tild, random_s) + rho * noise
            y_repeated = np.repeat(y[:, np.newaxis], repeats=num_of_combinations, axis=1)

            # calculate H_tild*s (mapped combinations at Rx)
            mapped_combinations = np.matmul(H_tild_repeated, combinations_repeated)

            # calculate ML term
            ml_term = np.linalg.norm(y_repeated - mapped_combinations, axis=2)**2

            # estimate symbols - minimize ML term
            s_estimation_indices = np.argmin(ml_term, axis=1)
            s_estimation = np.squeeze(combinations[s_estimation_indices])
            
            # calculate SER
            curr_SER = hlp.calculate_SER(s_estimation[:,:,np.newaxis], random_s, REP*M)
            SER_array_SM[i].append(curr_SER)
        plt.plot(snr_array_dB, SER_array_SM[i], label=f"SM with ML {N}Rx X {M}Tx")

    # STC 2x1 (copied and modified from Lecture_2_Ex2.py)
    N = 1
    for snr in snr_array:
        rho = 1/np.sqrt(snr)

        # generate symbols
        random_s0 = np.random.choice(symbols_list, (REP, 1))
        random_s1 = np.random.choice(symbols_list, (REP, 1))
        s_arr = np.concatenate((random_s0, random_s1), axis=1)
        s_arr = s_arr.reshape(REP, 2, 1)

        # generate channels and noise
        h0 = (np.random.standard_normal((REP, 1)) + 1j*np.random.standard_normal((REP, 1)))/2**0.5
        h1 = (np.random.standard_normal((REP, 1)) + 1j*np.random.standard_normal((REP, 1)))/2**0.5

        if N == 1:
            H_mat = create_H(h0, h1)

        elif N == 2:
            g0 = (np.random.standard_normal((REP, 1)) + 1j*np.random.standard_normal((REP, 1)))/2**0.5
            g1 = (np.random.standard_normal((REP, 1)) + 1j*np.random.standard_normal((REP, 1)))/2**0.5

            H_mat = create_H(h0, h1, g0, g1)
        
        n0 = (np.random.standard_normal((REP, N)) + 1j*np.random.standard_normal((REP, N)))/2**0.5
        n1 = (np.random.standard_normal((REP, N)) + 1j*np.random.standard_normal((REP, N)))/2**0.5
        n_arr = np.concatenate((n0, n1), axis=1)
        n_arr = n_arr.reshape(REP, N*2, 1)

        # calculate y
        y_arr = np.matmul(H_mat, s_arr) * 1/2**0.5 + rho*n_arr

        # calculate s_hat from the columns of H

        H0 = H_mat[:,:,0][:,:,np.newaxis] # first column
        H1 = H_mat[:,:,1][:,:,np.newaxis] # second column

        H0_conj = np.conj(H0)
        s0_hat = np.sum(H0_conj * y_arr, axis=(1, 2))/np.linalg.norm(H0, axis=(1,2))**2

        H1_conj = np.conj(H1)
        s1_hat = np.sum(H1_conj * y_arr, axis=(1, 2))/np.linalg.norm(H1, axis=(1,2))**2

        # estimate symbols
        s0_estimation = hlp.estimate_closest_symbols(s0_hat, symbols_list)
        s1_estimation = hlp.estimate_closest_symbols(s1_hat, symbols_list)

        # calculate SER
        SER_s0 = hlp.calculate_SER(s0_estimation[:,np.newaxis], random_s0, REP)
        SER_s1 = hlp.calculate_SER(s1_estimation[:,np.newaxis], random_s1, REP)
        SER_total = (SER_s0 + SER_s1)/2
        SER_array_STC.append(SER_total)
    plt.plot(snr_array_dB, SER_array_STC, label=f"STC 2X{N}")

    plt.legend()
    plt.grid(which='both', axis='both')
    plt.xticks(snr_array_dB)
    plt.yscale('log')
    plt.title(f'SER vs SNR - SM with ML detection (2x2) vs STC (2x1)\n{REP} repetitions')
    plt.ylabel('SER')
    plt.xlabel('SNR[dB]')
    plt.show()

if __name__ == "__main__":
    main()
    breakpoint()