import mne
import pandas as pd
import numpy as np


def load_eeg_data(file_path, channel_name):
    raw = mne.io.read_raw_edf(file_path)
    raw.pick_channels([channel_name,])
    return np.array(raw.to_data_frame()[channel_name])


def load_annotations(file_path):
    df = pd.read_csv(file_path)
    time_index = np.array(df['Onset'])
    time_index = time_index[:-1]
    label = np.array(df['Annotation'])
    label = label[:-1]
    label_map = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2,
                 'Sleep stage 3': 3, 'Sleep stage 4': 4, 'Sleep stage R': 5}
    for i in range(label.shape[0]):
        label[i] = label_map[label[i]]
    return time_index[1:].astype('uint32')*100,label[1:].astype('uint32')

def load_data(eeg_path,annotation_path,channel):
    data = load_eeg_data(eeg_path,channel)
    time,annotation = load_annotations(annotation_path)
    data = data[time[0]:time[-1]]
    time-=time[0]
    return data,time,annotation

def split_data(data,time,annotation):
    data = np.reshape(data,(-1,3000))
    label = []
    time = np.diff(time)
    for i,j in zip(time,annotation):
        label += [j,]*int(i/3000)
    return data,np.array(label)


if __name__ == '__main__':
    data,time,annotation = load_data('data\\edf_data\\SC4001E0-PSG.edf','data/csv_data/SC4001E0-PSG_annotations.csv','EEG Pz-Oz')
    data,label = split_data(data,time,annotation)
    print(data.shape,label)