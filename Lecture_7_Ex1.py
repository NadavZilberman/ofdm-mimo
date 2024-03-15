import matplotlib.pyplot as plt
import numpy as np

def main():

    # jaked calculation
    V = 120/3.6
    fc = 2.4e9
    f_max = fc*V/3e8
    fs = f_max*40
    FFT_size = 8192
    freq_linspace = np.arange(-FFT_size/2, FFT_size/2) / FFT_size * fs
    Sf = 1/(f_max*np.sqrt(1-(freq_linspace/f_max)**2))
    Sf[np.isnan(Sf)] = 0
    Sf = Sf/np.sum(Sf)

    Gf = Sf**.5

    gt = np.fft.fftshift(np.fft.ifft(Gf))
    gt = gt / np.linalg.norm(gt)

    # applying jaked PSD and power
    N = 100000 + FFT_size
    taps_delay = np.array([0, 310, 710, 1090, 1730, 2510]) * 1e-9
    taps_power_dB = np.array([0, -1, -9, -10, -15, -20])
    taps_power = 10**(taps_power_dB/10)

    gt_response = []

    for power in taps_power:
        wn = (np.random.standard_normal((N, )) + 1j*np.random.standard_normal((N, )))/2**0.5
        jaked = np.convolve(wn, gt) * power
        gt_response.append(jaked)

    gt_response = np.array(gt_response)
    # plt.plot(gt_response.T)

    coherence_time = 1/(5*f_max)
    t0_time = 0.5
    delta_1 = coherence_time / 5
    delta_2 = coherence_time * 10
    t0 = int(fs*t0_time)
    t1 = int(fs*(t0_time+delta_1))
    t2 = int(fs*(t0_time+delta_2))
    # calculating frequency response in a t0
    freq_linspace_to_show = np.arange(-10e6, 10e6, 1e3)
    channel_signal = np.zeros_like(freq_linspace_to_show, dtype=np.complex128)
    for i, delay in enumerate(taps_delay):
        channel_signal += gt_response[i,t0] * np.exp(-1j*2*np.pi*delay*freq_linspace_to_show)
    plt.plot(np.abs(channel_signal), label="t0")

    # calculating frequency response in a t0 + short time
    channel_signal = np.zeros_like(freq_linspace_to_show, dtype=np.complex128)
    for i, delay in enumerate(taps_delay):
        channel_signal += gt_response[i,t1] * np.exp(-1j*2*np.pi*delay*freq_linspace_to_show)
    plt.plot(np.abs(channel_signal), label="t0 + short time")

    # calculating frequency response in a t0 + long time
    channel_signal = np.zeros_like(freq_linspace_to_show, dtype=np.complex128)
    for i, delay in enumerate(taps_delay):
        channel_signal += gt_response[i,t2] * np.exp(-1j*2*np.pi*delay*freq_linspace_to_show)
    plt.plot(np.abs(channel_signal),  label="t0 + long time")

    plt.legend()
    plt.title(f"Magnitude of the channel: time t0 , t1 = t0 + short time, t2=t0 + long time\n(compared to coherence time)\nV = {V:.2f}m/s, fc = {fc/1e9}GHz")
    plt.xlabel("Frequency")
    plt.ylabel("|H(f)|")
    plt.grid()
    plt.yscale("log")
    plt.show()
    pass

if __name__ == "__main__":
    main()