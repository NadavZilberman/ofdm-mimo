import matplotlib.pyplot as plt
import numpy as np



def main():
    num_of_angels = 1000
    doa_linspace = np.linspace(-90, 90, num_of_angels)

    desired_doa_array = [-30, 30, 60]
    num_of_rx_array = [4, 16]

    doa_linspace_rad = np.radians(doa_linspace)
    
    for n_rx in num_of_rx_array:
        plt.figure()
        for desired_doa in desired_doa_array:
            response_abs = np.abs(np.sum([np.exp(1j*np.pi*n*(np.sin(np.radians(desired_doa)) - np.sin(doa_linspace_rad)))/n_rx for n in range(n_rx)], axis=0))
            response_abs = 10*np.log10(response_abs)
            plt.plot(doa_linspace, response_abs, label=f"Desired at {desired_doa}deg")

        plt.legend()
        plt.grid(which='both', axis='both')
        plt.xticks(np.arange(-90, 100, 10))
        plt.title(f"MRC Response with {n_rx} Rx antennas, for different desired DoAs")
        plt.ylim(-32, 2)
        plt.ylabel("Response(dB)")
        plt.xlabel("DoA(deg)")

    plt.show()
    pass
if __name__ == "__main__":
    main()