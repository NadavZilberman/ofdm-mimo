import matplotlib.pyplot as plt
import numpy as np
import helpers as hlp

def main():
    plt.figure()
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5
    snr_array_dB = np.arange(0, 41)
    snr_array = 10**(snr_array_dB/10)
    REP = 500000
    ant_pairs = [[2,2], [2,4]] # [M, N]
    SER_array_SM = [[], []] # Reminder: make sure there's the same number of sub arrays
    SER_array_ZF = [[], []] # Reminder: make sure there's the same number of sub arrays
    
    # SM with ML (copied and modified from Lecture_4_Ex1.py)

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
        plt.plot(snr_array_dB, SER_array_SM[i], label=f"SM with ML {M}Tx X {N}Rx")

    # Zero Forcing
    
    for i, [M, N] in enumerate(ant_pairs):
        for snr in snr_array:
            rho = 1/np.sqrt(snr)

            # generate symbols
            random_s = np.random.choice(symbols_list, (REP, M, 1))
            
            # generate channels and noise
            H = (np.random.standard_normal((REP, N, M)) + 1j*np.random.standard_normal((REP, N, M)))/2**0.5
            noise = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5

            # calculate H_tild
            H_tild = H/M**0.5
            
            # calculate y and repeat it
            y = np.matmul(H_tild, random_s) + rho * noise

            # calculate s_hat
            H_tild_conj = np.transpose(np.conj(H_tild), (0,2,1))
            s_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(H_tild_conj, H_tild)), H_tild_conj), y)
            
            s_hat = np.squeeze(s_hat, axis=2)
            s_hat = np.reshape(s_hat, (REP*M,))

            # estimate symbols
            s_estimation = hlp.estimate_closest_symbols(s_hat, symbols_list)
            s_estimation = np.reshape(s_estimation, (REP, M))[:,:,np.newaxis] # modify shape for calculate_SER()

            # calculate SER
            SER = hlp.calculate_SER(s_estimation, random_s, REP*M) # don't forget the *M
            SER_array_ZF[i].append(SER) 
        plt.plot(snr_array_dB, SER_array_ZF[i], label=f"ZF {M}Tx X {N}Rx")

    plt.legend()
    plt.grid(which='both', axis='both')
    plt.xticks(snr_array_dB)
    plt.yscale('log')
    plt.title(f'SER vs SNR - SM with ML detection vs ZF - (2Tx X 2Rx) and (2Tx X 4Rx)\n{REP} repetitions')
    plt.ylabel('SER')
    plt.xlabel('SNR[dB]')
    plt.show()
    
if __name__ == "__main__":
    main()
    breakpoint()