import matplotlib.pyplot as plt
import numpy as np
import helpers as hlp
from Lecture_2_Ex2 import create_H

def main():
    plt.figure()
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5
    snr_array_dB = np.arange(0, 16)
    snr_array = 10**(snr_array_dB/10)
    REP = 1000000
    ant_pairs = [[2,2], [4,2]]
    SER_array_BF = [[], []]
    SER_array_STC = []

    for i, [M, N] in enumerate(ant_pairs):
        for snr in snr_array:
            rho = 1/np.sqrt(snr)

            # generate symbols
            random_s = np.random.choice(symbols_list, (REP, 1))

            # generate channels and noise
            H = (np.random.standard_normal((REP, N, M)) + 1j*np.random.standard_normal((REP, N, M)))/2**0.5
            noise = (np.random.standard_normal((REP, N, 1)) + 1j*np.random.standard_normal((REP, N, 1)))/2**0.5
            
            # calculate optimal modulator at the transmitter
            U, D, Vh = np.linalg.svd(H)
            V = np.transpose(np.conj(Vh), (0, 2, 1))
            w = V[:,:,0][:,:, np.newaxis]

            # calculate y
            Hw = np.matmul(H, w)
            y = Hw * random_s[:, np.newaxis, :] + rho*noise

            # calculate s_hat
            Hw_conj = np.transpose(np.conj(Hw), (0, 2, 1))
            s_hat = np.matmul(Hw_conj, y)/np.linalg.norm(Hw, axis=(1,2))[:, np.newaxis, np.newaxis]**2 # TODO check
            s_hat = np.squeeze(s_hat, axis=(1,2))

            # estimate symbols
            s_estimation = hlp.estimate_closest_symbols(s_hat, symbols_list)

            # calculaate SER
            curr_SER = hlp.calculate_SER(s_estimation[:,np.newaxis], random_s, REP)
            SER_array_BF[i].append(curr_SER)
        plt.plot(snr_array_dB, SER_array_BF[i], label=f"Eigen BF {N}Rx X {M}Tx")


    # STC 2x2 (copied and modified from Lecture_2_Ex2.py)
        
    N = 2 # Rx ant

    for snr in snr_array:
        curr_ser = 0
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
    plt.title(f'SER vs SNR - Eigen BF (2x2), (2x4) vs STC (2x2)\n{REP} repetitions')
    plt.ylabel('SER')
    plt.xlabel('SNR[dB]')
    plt.show()

    # As we've seen in class, the E-BF 2x2 is ~2.3dB better then STC 2x2
    # AG_BF/AG_STC = 3.36/N = 3.36/2 = 1.68 = 2.25dB

if __name__ == "__main__":
    main()
    breakpoint()