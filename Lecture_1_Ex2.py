import matplotlib.pyplot as plt
import numpy as np
import helpers as hlp

def main():
    m = 10
    REP = 5000000
    f = 2.5e9

    real_vals = []
    imag_vals = []

    for n in range(REP):
        taui = np.random.uniform(50e-3, 100e-3, m)/(3e8*1e-3)

        # knowing that F{delta(tau-taui)}(f) = exp(-2*pi*1j*f*taui):

        Hi = np.exp(-2*np.pi*1j*f*taui)
        H = np.sum(Hi)

        H_real = np.real(H)
        H_imag = np.imag(H)
        real_vals.append(H_real)
        imag_vals.append(H_imag)

    real_vals_normalized = real_vals/np.sum(real_vals)
    imag_vals_normalized = imag_vals/np.sum(imag_vals)

    correlation = np.correlate(real_vals_normalized, imag_vals_normalized)
    var_real = np.var(real_vals)
    var_imag = np.var(imag_vals)
    print("mean(real) = {}, var(real) = {}".format(np.mean(real_vals), np.var(real_vals)))
    print("mean(imag) = {}, var(imag) = {}".format(np.mean(imag_vals), np.var(imag_vals)))
    print("correlation is ", correlation)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.hist(real_vals, 1024)
    ax1.set_xlabel("bins")
    ax1.set_title("PDF of real(H)")
    ax1.set_ylabel("number")
    ax1.grid()
    ax2.hist(imag_vals, 1024)
    ax2.set_xlabel("bins")
    ax2.set_title("PDF of imag(H)")
    ax2.set_ylabel("number")
    ax2.grid()

    # abs_H = np.abs(real_vals+1j*np.ones((N))*imag_vals)
    # abs_H_normalized = abs_H/sum(abs_H)
    # ax3.hist(abs_H_normalized, 1024)
    # ax3.set_xlabel("bins")
    # ax3.set_title("abs(H)")
    # ax3.set_ylabel("number")
    # ax3.grid()
    plt.show()
    pass

if __name__ == "__main__":
    main()