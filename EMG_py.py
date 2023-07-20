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

def EMG_plot_noise(x_size, y_size, emg_signal_healthy_range, emg_signal_myopathy_range, emg_signal_neuropathy_range,fs, start):#Zaszumiony sygnał

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


main_plot()



