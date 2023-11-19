import matplotlib.pyplot as plt
import configargparse
import json
import numpy as np
import os

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--spike_path',default='./data/spike/')
    parser.add_argument('--image_save_path',default='./output/images/')
    parser.add_argument('--accurancy_path',default='./output/exp100/')
    return parser

def plot_spike(args):
    try:
        os.mkdir(args.image_save_path+'origional_spike/')
    except:
        pass
    with open(args.spike_path+'origional_spike.json','r') as f:
        spike_data = json.load(f)
    for key in spike_data.keys():
        band_data = spike_data[key]
        figure = plt.figure()
        for i,key1 in enumerate(band_data.keys()):
            subplot = figure.add_subplot(5,1,i+1)
            spike_up = np.array(band_data[key1]['up'])
            spike_dn = np.array(band_data[key1]['dn'])
            subplot.bar(np.concatenate([spike_up,spike_dn]),np.concatenate([np.ones_like(spike_up),-np.ones_like(spike_dn)]),0.05,color='k')
            # subplot.stem(np.concatenate([spike_up,spike_dn]),np.concatenate([np.ones_like(spike_up),-np.ones_like(spike_dn)]),markerfmt=' ',linefmt='k-')
            subplot.set_xlim((0.,30.))
            subplot.set_title(key1)
        figure.savefig(args.image_save_path+'origional_spike/'+key+'.png')

def plot_snn_spike(args):
    try:
        os.mkdir(args.image_save_path+'snn_spike/')
    except:
        pass
    with open(args.spike_path+'snn_spike.json','r') as f:
        spike_data = json.load(f)
    for key in spike_data.keys():
        band_data = spike_data[key]
        figure = plt.figure()
        for i,key1 in enumerate(band_data.keys()):
            subplot = figure.add_subplot(5,1,i+1)
            spike = np.array(band_data[key1])
            subplot.plot(spike,np.zeros_like(spike),'k.',markersize=1)
            subplot.set_xlim((0.,30.))
            subplot.set_title(key1)
            if i == 4:
                break
        figure.savefig(args.image_save_path+'snn_spike/'+key+'.png')

def plot_accurancy(args):
    with open(args.accurancy_path+'acc.json','r') as f:
        data = json.load(f)
        train_acc = data['train']
        test_acc = data['test']
        times = data['time']
    plt.plot(times,train_acc,label='train')
    plt.plot(times,test_acc,label='test')
    plt.legend()
    plt.savefig(args.accurancy_path+'acc.png')

if __name__ == '__main__':
    args = config_parser().parse_args()
    # plot_spike(args)
    # plot_snn_spike(args)
    plot_accurancy(args)