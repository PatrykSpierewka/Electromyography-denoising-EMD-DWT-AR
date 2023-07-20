import pywt
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import pyhht
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import lfilter
from sklearn.metrics import mean_squared_error, mean_absolute_error

def Load_data(start, stop):

    #healthy person
    record_healthy = wfdb.rdrecord('emg_healthy', channels=[0])
    #signal flattening
    emg_signal_healthy = record_healthy.p_signal.flatten()
    #reducing the signal range
    emg_signal_healthy_range = emg_signal_healthy[start : stop]

    #person with myopathies
    record_myopathy = wfdb.rdrecord('emg_myopathy', channels=[0])
    emg_signal_myopathy = record_myopathy.p_signal.flatten()
    emg_signal_myopathy_range = emg_signal_myopathy[start : stop]

    #suffering from neuropathy
    record_neuropathy = wfdb.rdrecord('emg_neuropathy', channels = [0])
    emg_signal_neuropathy = record_neuropathy.p_signal.flatten()
    emg_signal_neuropathy_range = emg_signal_neuropathy[start : stop]

    return emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, start

def EMG_plot_noise(x_size, y_size, emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range,fs, start):

    offset = start/fs
    time = np.arange(len(emg_signal_healthy_range)) / fs + offset
    plt.figure(figsize=(x_size, y_size))

    #Subplot for a healthy person
    plt.subplot(311)
    plt.plot(time, emg_signal_healthy_range, label='Noisy EMG signal of a healthy person')
    plt.title('EMG signal before denoising')
    plt.ylabel('Amplitude [mV]')
    plt.legend()

    #Subplot for a person with myopathy
    plt.subplot(312)
    plt.plot(time, emg_signal_myopathy_range, label='Noisy EMG signal of a person with myopathy')
    plt.ylabel('Amplitude [mV]')
    plt.legend()

    #Subplot for a person with neuropathy
    plt.subplot(313)
    plt.plot(time, emg_signal_neuropathy_range, label='Noisy EMG signal of a person with neuropathy')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mV]')
    plt.legend()
    plt.show()

#Display
def main_plot():
    emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, start = Load_data(0, 8000)
    EMG_plot_noise(19, 15, emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, 4000, start)


'''
main_plot()
'''


#Empirical Mode Decomposition denoising
def EMD(emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range):#Empirical mode decomposition

    decomposer_healthy = pyhht.EMD(emg_signal_healthy_range) #An object is created, which will be used further for decomposition into IMFs
    imfs_healthy = decomposer_healthy.decompose() #Decomposition of the EMG signal into its IMF harmonic components.
    noise_window_healthy = emg_signal_healthy_range[0:500] #Selection of samples in the signal interval [a,b] representing the noise
    thresholds_healthy = np.std(noise_window_healthy)
    imfs_healthy_filtered = [pywt.threshold(c, thresholds_healthy) for c in imfs_healthy] #Thresholding
    emg_signal_healthy_denoised_EMD = np.sum(imfs_healthy_filtered, axis=0) #Reconstruction of the denoised signal by summing the denoised IMF

    decomposer_myopathy = pyhht.EMD(emg_signal_myopathy_range)
    noise_window_myopathy = emg_signal_myopathy_range[2000:3000]
    thresholds_myopathy = np.std(noise_window_myopathy)
    imfs_myopathy = decomposer_myopathy.decompose()
    imfs_myopathy_filtered = [pywt.threshold(c, thresholds_myopathy) for c in imfs_myopathy]
    emg_signal_myopathy_denoised_EMD = np.sum(imfs_myopathy_filtered, axis=0)

    decomposer_neuropathy = pyhht.EMD(emg_signal_neuropathy_range)
    imfs_neuropathy = decomposer_neuropathy.decompose()
    noise_window_neuropathy = emg_signal_neuropathy_range[0:1000]
    thresholds_neuropathy = np.std(noise_window_neuropathy)
    imfs_neuropathy_filtered = [pywt.threshold(c, thresholds_neuropathy) for c in imfs_neuropathy]
    emg_signal_neuropathy_denoised_EMD = np.sum(imfs_neuropathy_filtered, axis=0)

    return emg_signal_healthy_denoised_EMD, emg_signal_myopathy_denoised_EMD, emg_signal_neuropathy_denoised_EMD

def EMG_plot_comparision_EMD(x_size, y_size, emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, emg_signal_healthy_denoised_EMD, emg_signal_myopathy_denoised_EMD, emg_signal_neuropathy_denoised_EMD, fs, start):
    offset = start/fs
    time = np.arange(len(emg_signal_healthy_range)) / fs + offset

    plt.figure(figsize=(x_size, y_size))

    #Subplot for a healthy person
    plt.subplot(311)
    plt.plot(time, emg_signal_healthy_range, label='Noisy EMG signal of a healthy person')
    plt.plot(time, emg_signal_healthy_denoised_EMD, label='Denoised EMG signal of a healthy person')
    plt.title('EMG signal after EMD denoising')
    plt.ylabel('Amplitude [mV]')
    plt.legend()

    #Subplot for a person with myopathy
    plt.subplot(312)
    plt.plot(time, emg_signal_myopathy_range, label='Noisy EMG signal of a person with myopathy')
    plt.plot(time, emg_signal_myopathy_denoised_EMD, label='Denoised EMG signal of a person with myopathy')
    plt.ylabel('Amplitude [mV]')
    plt.legend()

    #Subplot for a person with neuropathy
    plt.subplot(313)
    plt.plot(time, emg_signal_neuropathy_range, label='Noisy EMG signal of a person with neuropathy')
    plt.plot(time, emg_signal_neuropathy_denoised_EMD, label='Denoised EMG signal of a person with neuropathy')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mV]')
    plt.legend()
    plt.show()

def main_EMD():
    emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, start = Load_data(0, 8000)
    emg_signal_healthy_denoised_EMD, emg_signal_myopathy_denoised_EMD, emg_signal_neuropathy_denoised_EMD = EMD(emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range)
    EMG_plot_comparision_EMD(19, 15, emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, emg_signal_healthy_denoised_EMD, emg_signal_myopathy_denoised_EMD, emg_signal_neuropathy_denoised_EMD, 4000, start)


'''
main_EMD()
'''


#Discrete wavelet transform denoising
def DWT(emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, wavelet, level, value, mode):

    coefficients_healthy = pywt.wavedec(emg_signal_healthy_range, wavelet, level = level)#Wavelet decomposition
    coefficients_healthy[1:] = (pywt.threshold(i, value = value, mode = mode) for i in coefficients_healthy[1:])#Thresholding in soft mode
    emg_signal_healthy_denoised_DWT = pywt.waverec(coefficients_healthy, wavelet) #Inverse wavelet transform using reduced wavelets

    #Wavelet transform denoising for a person suffering from myopathy
    coefficients_myopathy = pywt.wavedec(emg_signal_myopathy_range, wavelet, level = level)
    coefficients_myopathy[1:] = (pywt.threshold(i, value = value, mode = mode) for i in coefficients_myopathy[1:])
    emg_signal_myopathy_denoised_DWT = pywt.waverec(coefficients_myopathy, wavelet)

    #Wavelet transform denoising for a person suffering from neuropathy
    coefficients_neuropathy = pywt.wavedec(emg_signal_neuropathy_range, wavelet, level = level)
    coefficients_neuropathy[1:] = (pywt.threshold(i, value = value, mode = mode) for i in coefficients_neuropathy[1:])
    emg_signal_neuropathy_denoised_DWT = pywt.waverec(coefficients_neuropathy, wavelet)

    return emg_signal_healthy_denoised_DWT, emg_signal_myopathy_denoised_DWT, emg_signal_neuropathy_denoised_DWT

def EMG_plot_comparision_DWT(x_size, y_size, emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, emg_signal_healthy_denoised_DWT, emg_signal_myopathy_denoised_DWT, emg_signal_neuropathy_denoised_DWT, fs, start):
    offset = start/fs
    time = np.arange(len(emg_signal_healthy_range)) / fs + offset

    plt.figure(figsize=(x_size, y_size))

    #Subplot for a healthy person
    plt.subplot(311)
    plt.plot(time, emg_signal_healthy_range, label='Noisy EMG signal of a healthy person')
    plt.plot(time, emg_signal_healthy_denoised_DWT, label='Denoised EMG signal of a healthy person')
    plt.title('EMG signal after DWT denoising')
    plt.ylabel('Amplitude [mV]')
    plt.legend()

    #Subplot for a person with myopathy
    plt.subplot(312)
    plt.plot(time, emg_signal_myopathy_range, label='Noisy EMG signal of a person with myopathy')
    plt.plot(time, emg_signal_myopathy_denoised_DWT, label='Denoised EMG signal of a person with myopathy')
    plt.ylabel('Amplitude [mV]')
    plt.legend()

    #Subplot for a person with neuropathy
    plt.subplot(313)
    plt.plot(time, emg_signal_neuropathy_range, label='Noisy EMG signal of a person with neuropathy')
    plt.plot(time, emg_signal_neuropathy_denoised_DWT, label='Denoised EMG signal of a person with neuropathy')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [mV]')
    plt.legend()
    plt.show()

def main_DWT():
    emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, start = Load_data(0, 8000)
    emg_signal_healthy_denoised_DWT, emg_signal_myopathy_denoised_DWT, emg_signal_neuropathy_denoised_DWT = DWT(emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, 'db4', 4, 1, 'soft')
    EMG_plot_comparision_DWT(19, 15, emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range, emg_signal_healthy_denoised_DWT, emg_signal_myopathy_denoised_DWT, emg_signal_neuropathy_denoised_DWT, 4000, start)


main_DWT()


