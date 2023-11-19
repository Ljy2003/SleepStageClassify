from scipy.signal import butter,filtfilt
import numpy as np

def butter_bandpass_filter(data, lowcut, highcut, fs=100, order=5):
    fa = 0.5 * fs
    low = lowcut / fa
    high = highcut / fa
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def get_eeg_band(signal):
    eeg_band = {}
    eeg_band['delta'] = butter_bandpass_filter(signal,1,4)
    eeg_band['theta'] = butter_bandpass_filter(signal,5,8)
    eeg_band['alpha'] = butter_bandpass_filter(signal,9,12)
    eeg_band['sigma'] = butter_bandpass_filter(signal,13,16)
    eeg_band['beta'] = butter_bandpass_filter(signal,17,25)
    return eeg_band


if __name__ == '__main__':
    import load_data
    import matplotlib.pyplot as plt

    def get_fft(signal):
        x = np.fft.fftshift(np.fft.fft(signal))
        x = np.abs(x)
        return x

    data,time,annotation = load_data.load_data('data\\edf_data\\SC4001E0-PSG.edf','data/csv_data/SC4001E0-PSG_annotations.csv','EEG Pz-Oz')
    data = data[0:3000]
    band = get_eeg_band(data)
    plt.subplot(6,1,1)
    plt.plot(get_fft(data))
    for i,key in enumerate(band.keys()):
        plt.subplot(6,1,i+2)
        plt.plot(get_fft(band[key]))
    plt.show()
    