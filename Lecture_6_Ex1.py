import matplotlib.pyplot as plt
import numpy as np
import helpers as hlp

def main():
    plt.figure()
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5
    snr_array_dB = np.arange(0, 26)
    snr_array = 10**(snr_array_dB/10)
    REP = 500000
    [M, N] = [2, 2]
    SER_array_SM = [[] ,[], []] # Three lists: for both symbols, s0 and s1
    SER_array_ZF = [[], [], []]
    
    # SM with ML, SVD precoding (copied and modified from Lecture_4_Ex1.py)

    combinations = np.array([[x, y] for x in symbols_list for y in symbols_list]).reshape(-1, M, 1)
    combinations_repeated = np.tile(combinations, (REP, 1, 1, 1))
    num_of_combinations = len(combinations)
    for snr in snr_array:
        rho = 1/np.sqrt(snr)

        # generate symbols
        random_s = np.random.choice(symbols_list, (REP, M, 1))

        # generate channels and noise
        H = (np.random.standard_normal((REP, N, M)) + 1j*np.random.standard_normal((REP, N, M)))/2**0.5
        noise = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5

        # svd precoding 
        U, D, Vh = np.linalg.svd(H)
        V = np.transpose(np.conj(Vh), (0, 2, 1))
        x = (1/M**0.5) * np.matmul(V, random_s)

        # calculate H_tild and repeat it
        HV = np.matmul(H, V)
        HV_tild = HV*(1/M**0.5)
        H_tild_repeated = np.repeat(HV_tild[:, np.newaxis], repeats=num_of_combinations, axis=1) # now we compare to H*V

        # calculate y and repeat it
        y = np.matmul(H, x) + rho * noise
        y_repeated = np.repeat(y[:, np.newaxis], repeats=num_of_combinations, axis=1)

        # calculate H_tild*s (mapped combinations at Rx)
        mapped_combinations = np.matmul(H_tild_repeated, combinations_repeated)

        # calculate ML term
        ml_term = np.linalg.norm(y_repeated - mapped_combinations, axis=2)**2

        # estimate symbols - minimize ML term
        s_estimation_indices = np.argmin(ml_term, axis=1)
        s_estimation = np.squeeze(combinations[s_estimation_indices])
        
        # calculate SER separately (only for two symbols)
        s_0_estimation = s_estimation[:,0]
        s_1_estimation = s_estimation[:,1]
        curr_SER_0 = hlp.calculate_SER(s_0_estimation[:,np.newaxis], random_s[:,0,:], REP)
        curr_SER_1 = hlp.calculate_SER(s_1_estimation[:,np.newaxis], random_s[:,1,:], REP)
        curr_SER = hlp.calculate_SER(s_estimation[:,:,np.newaxis], random_s, REP*M)
        SER_array_SM[0].append(curr_SER)
        SER_array_SM[1].append(curr_SER_0)
        SER_array_SM[2].append(curr_SER_1)
    plt.plot(snr_array_dB, SER_array_SM[0], label=f"SM with ML and SVD precoding {M}Tx X {N}Rx")


    # Zero Forcing, SVD precoding (copied and modified from Lecture_4_Ex2.py)
    
    for snr in snr_array:
        rho = 1/np.sqrt(snr)

        # generate symbols
        random_s = np.random.choice(symbols_list, (REP, M, 1))
        
        # generate channels and noise
        H = (np.random.standard_normal((REP, N, M)) + 1j*np.random.standard_normal((REP, N, M)))/2**0.5
        noise = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5

        U, D, Vh = np.linalg.svd(H)
        V = np.transpose(np.conj(Vh), (0, 2, 1))
        x = (1/2**0.5) * np.matmul(V, random_s)

        # calculate H_tild
        HV = np.matmul(H, V)
        HV_tild = HV/M**0.5
        H_tild = H/M**0.5

        # calculate y and repeat it
        y = np.matmul(H, x) + rho * noise

        # calculate s_hat
        # H_tild_conj = np.transpose(np.conj(HV), (0,2,1))
        H_tild_conj = np.transpose(np.conj(H), (0,2,1))
        # s_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(H_tild_conj, HV)), H_tild_conj), y)
        s_hat = np.matmul(np.linalg.inv(V), np.matmul(np.matmul(np.linalg.inv(np.matmul(H_tild_conj, H)), H_tild_conj), y))
        
        s_hat = np.squeeze(s_hat, axis=2)
        s_hat = np.reshape(s_hat, (REP*M,))

        # estimate symbols
        s_estimation = hlp.estimate_closest_symbols(s_hat, symbols_list)
        s_estimation = np.reshape(s_estimation, (REP, M))[:,:,np.newaxis] # modify shape for calculate_SER()

        # calculate SER separately (only for two symbols)
        s_0_estimation = s_estimation[:,0][:]
        s_1_estimation = s_estimation[:,1]
        curr_SER_0 = hlp.calculate_SER(s_0_estimation, random_s[:,0,:], REP)
        curr_SER_1 = hlp.calculate_SER(s_1_estimation, random_s[:,1,:], REP)
        curr_SER = hlp.calculate_SER(s_estimation, random_s, REP*M) # don't forget the *M
        SER_array_ZF[0].append(curr_SER)
        SER_array_ZF[1].append(curr_SER_0)
        SER_array_ZF[2].append(curr_SER_1)
    plt.plot(snr_array_dB, SER_array_ZF[0], label=f"ZF with SVD precoding {M}Tx X {N}Rx")

    plt.legend()
    plt.grid(which='both', axis='both')
    plt.xticks(snr_array_dB)
    plt.yscale('log')
    plt.title(f'SER vs SNR - SM with ML detection vs ZF - SVD precoding (2Tx X 2Rx)\n{REP} repetitions')
    plt.ylabel('SER')
    plt.xlabel('SNR[dB]')
    plt.show()

    # stream 1 plot
    plt.figure()
    plt.plot(snr_array_dB, SER_array_SM[1], label=f"SM with ML and SVD precoding {M}Tx X {N}Rx - symbol 1")
    plt.plot(snr_array_dB, SER_array_ZF[1], label=f"ZF with SVD precoding {M}Tx X {N}Rx - symbol 1")

    plt.legend()
    plt.grid(which='both', axis='both')
    plt.xticks(snr_array_dB)
    plt.yscale('log')
    plt.title(f'SER vs SNR - SM with ML detection vs ZF - SVD precoding (2Tx X 2Rx) - Symbol 1\n{REP} repetitions')
    plt.ylabel('SER')
    plt.xlabel('SNR[dB]')
    plt.show()

    # stream 2 plot
    plt.figure()
    plt.plot(snr_array_dB, SER_array_SM[2], label=f"SM with ML and SVD precoding {M}Tx X {N}Rx - symbol 2")
    plt.plot(snr_array_dB, SER_array_ZF[2], label=f"ZF with SVD precoding {M}Tx X {N}Rx - symbol 2")

    plt.legend()
    plt.grid(which='both', axis='both')
    plt.xticks(snr_array_dB)
    plt.yscale('log')
    plt.title(f'SER vs SNR - SM with ML detection vs ZF - SVD precoding (2Tx X 2Rx) - Symbol 2\n{REP} repetitions')
    plt.ylabel('SER')
    plt.xlabel('SNR[dB]')
    plt.show()

if __name__ == "__main__":
    main()
    breakpoint()