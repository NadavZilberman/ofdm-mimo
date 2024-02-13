import matplotlib.pyplot as plt
import numpy as np
import helpers as hlp

def create_H(h0, h1, g0=None, g1=None):
    # *** create H *** #
    if g0 is None:
        h0 = h0.reshape(-1, 1)
        h1 = h1.reshape(-1, 1)
        conj_h0 = np.conj(h0)
        conj_h1 = np.conj(h1)
        col1 = np.hstack((h0, conj_h1))
        col2 = np.hstack((h1, -conj_h0))
        H_mat = np.stack((col1, col2), axis=-1)
    else:
        h0 = h0.reshape(-1, 1)
        h1 = h1.reshape(-1, 1)
        conj_h0 = np.conj(h0)
        conj_h1 = np.conj(h1)
        g0 = g0.reshape(-1, 1)
        g1 = g1.reshape(-1, 1)
        conj_g0 = np.conj(g0)
        conj_g1 = np.conj(g1)
        col1 = np.hstack((h0, conj_h1, g0, conj_g1))
        col2 = np.hstack((h1, -conj_h0, g1, -conj_g0))
        H_mat = np.stack((col1, col2), axis=-1)
    # *** calculate inverse of H *** #
    # if g0 is None:
    #     H_resahped = H_mat.reshape(-1, 2, 2)
    #     H_inverse_reshaped = np.linalg.inv(H_resahped)
    #     H_inverse = H_inverse_reshaped.reshape(repetitions,  2, 2)
    # else:
    #     H_inverse = None
    return H_mat

def main():
    plt.figure()
    symbols_list = np.array([-1-1j, -1+1j, 1-1j, 1+1j])/2**0.5
    snr_array_dB = np.arange(0, 26)
    snr_array = 10**(snr_array_dB/10)
    REP = 5000000
    N = 1 # Rx ant
    SER_array_STC = []

    # STC 2x1

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


    # MRC 1X2 (copied from Lecture_2_Ex1.py)
    N = 2 
    SER_array_MRC = []
    random_symbols = np.random.choice(symbols_list, (REP, 1))

    for snr in snr_array:
        rho = 1/np.sqrt(snr)
        h = (np.random.standard_normal((REP, N)) + 1j*np.random.standard_normal((REP, N)))/2**0.5
        n = (np.random.standard_normal((REP, N)) + 1j*np.random.standard_normal((REP, N)))/2**0.5
        y = h * random_symbols + rho * n
        # calc s_hat
        s_hat = np.sum(np.conj(h[:,:,np.newaxis]) * y[:,:,np.newaxis], axis=(1, 2))/np.linalg.norm(h, axis=1)**2
        # # estimate symbols
        s_estimation = hlp.estimate_closest_symbols(s_hat, symbols_list)
        # calculate SER
        curr_SER = hlp.calculate_SER(s_estimation, random_symbols.T, REP)
        SER_array_MRC.append(curr_SER)

    plt.plot(snr_array_dB, SER_array_MRC, label=f"MRC 1x2")

    plt.legend()
    plt.grid(which='both', axis='both')
    plt.xticks(snr_array_dB)
    plt.yscale('log')
    plt.title(f'SER vs SNR - STC 2x1 vs MRC 1x2\n{REP} repetitions')
    plt.ylabel('SER')
    plt.xlabel('SNR[dB]')
    plt.show()
    pass
    # We see in the graph that MRC is better by 3dB.

if __name__ == "__main__":
    main()