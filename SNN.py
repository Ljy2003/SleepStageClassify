from scipy.interpolate import interp1d
import numpy as np
import utils.load_data as load_data
import matplotlib.pyplot as plt
import brian2 as b2
from brian2 import NeuronGroup, Synapses, Network
import utils.weights as weights
import configargparse
import json
import utils.filter as filter
import tqdm
import random

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        '--eeg_path', default='data\\edf_data\\SC4001E0-PSG.edf')
    parser.add_argument('--annotation_path',
                        default='data/csv_data/SC4001E0-PSG_annotations.csv')
    parser.add_argument('--channel', default='EEG Pz-Oz')
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--spike_save_path', default='./data/spike/')
    parser.add_argument('--tau', type=int, default=0.1, help='units: second')
    parser.add_argument('--hidden_size',type=int,default=20)
    return parser
    # parser.parse_args()


def find_thresholds(signals, times, window_size, sample_ratio, scaling_factor):
    '''
    This functions retuns the mean threshold for your signals, based on the calculated
    mean noise floor and a user-specified scaling facotr that depeneds on the type of signals,
    characteristics of patterns, etc.

    Parameters
    -------
    signals : array
        amplitude of the signals
    times : array
        time vector
    window : float
        time window [same units as time vector] where the maximum amplitude
        of the signals will be calculated
    sample_ratio : float
        the percentage of time windows that will be used to
        calculate the mean maximum amplitude.
     scaling_factor : float
        a percentage of the calculated threshold
    '''
    min_time = np.min(times)
    if np.min(times) < 0:
        raise ValueError(
            f'Tried to find thresholds for a dataset with a negative time: {min_time}')
    duration = np.max(times) - min_time
    if duration <= 0:
        raise ValueError(
            f'Tried to find thresholds for a dataset with a duration that under or equal to zero. Got duration: {duration}')

    if len(signals) == 0:
        raise ValueError('signals is not allowed to be empty, but was'
                         )
    if len(times) == 0:
        raise ValueError('times is not allowed to be empty, but was')

    if len(signals) != len(times):
        raise ValueError(
            f'signals and times need to have corresponding indices, but signals has length {len(signals)} while times has length {len(times)}')

    if not 0 < sample_ratio < 1:
        raise ValueError(
            f'sample_ratio must be a value between 0 and 1, but was {sample_ratio}'
        )

    num_timesteps = int(np.ceil(duration / window_size))
    max_min_amplitude = np.zeros((num_timesteps, 2))
    for interval_nr, interval_start in enumerate(np.arange(start=0, stop=duration, step=window_size)):
        interval_end = interval_start + window_size
        index = np.where((times >= interval_start) & (times <= interval_end))
        max_amplitude = np.max(signals[index])
        min_amplitude = np.min(signals[index])
        max_min_amplitude[interval_nr, 0] = max_amplitude
        max_min_amplitude[interval_nr, 1] = min_amplitude

    chosen_samples = max(int(np.round(num_timesteps * sample_ratio)), 1)
    threshold_up = np.mean(np.sort(max_min_amplitude[:, 0])[:chosen_samples])
    threshold_dn = np.mean(
        np.sort(max_min_amplitude[:, 1] * -1)[:chosen_samples])
    return scaling_factor*(threshold_up + threshold_dn)


def signal_to_spike(signal, times, threshold_up, threshold_down, interpolation_factor, refractory_period):
    '''
    This functions retuns two spike trains, when the signal crosses the specified threshold in
    a rising direction (UP spikes) and when it crosses the specified threshold in a falling
    direction (DOWN spikes)
    :times (array): time vector
    :amplitude (array): amplitude of the signal
    :interpolation_factor (int): upsampling factor, new sampling frequency
    :thr_up (float): threshold crossing in a rising direction
    :thr_dn (float): threshold crossing in a falling direction
    :refractory_period (float): period in which no spike will be generated [same units as time vector]
    '''
    actual_dc = 0
    spike_up = []
    spike_dn = []

    intepolated_time = interp1d(times, signal)
    rangeint = np.round(
        (np.max(times) - np.min(times))*interpolation_factor)
    xnew = np.linspace(np.min(times), np.max(
        times), num=int(rangeint), endpoint=True)
    data = np.reshape([xnew, intepolated_time(xnew)], (2, len(xnew))).T

    i = 0
    while i < (len(data)):
        if ((actual_dc + threshold_up) < data[i, 1]):
            spike_up.append(data[i, 0])  # spike up
            actual_dc = data[i, 1]        # update current dc value
            i += int(refractory_period *
                     interpolation_factor)
        elif ((actual_dc - threshold_down) > data[i, 1]):
            spike_dn.append(data[i, 0])  # spike dn
            actual_dc = data[i, 1]        # update curre
            i += int(refractory_period *
                     interpolation_factor)
        else:
            i += 1

    return np.array(spike_up), np.array(spike_dn)


def spike(data, interpolation_factor=35_000, refractory_period=0.01):
    time = np.arange(0, data.shape[0])*0.01
    threshold = np.ceil(find_thresholds(
        data, time, window_size=0.5, sample_ratio=1/6, scaling_factor=0.3))
    spike_up, spike_dn = signal_to_spike(
        data, time, threshold, threshold, interpolation_factor=interpolation_factor, refractory_period=refractory_period)
    return spike_up, spike_dn


def snn(tau,args):
    eq_LIF = '''
    dv/dt = -v/(1*second): 1
    tau :second
    '''

    net = {}
    net['input'] = b2.SpikeGeneratorGroup(
        2, indices=[0,], times=[0,]*b2.second,dt=0.01*b2.second)
    net['neu_hidden'] = NeuronGroup(args.hidden_size, eq_LIF, threshold='v>0.8', reset='v=0',method='euler',dt=0.01*b2.second)
    net['neu_hidden'].tau = tau*b2.second
    net['neu_output'] = NeuronGroup(1, eq_LIF, threshold='v>0.8', reset='v=0',method='euler',dt=0.01*b2.second)
    net['neu_output'].tau = tau*b2.second
    net['poisson_input'] = b2.PoissonGroup(1, 50*b2.Hz)
    net['global_inhibitory'] = NeuronGroup(
        1, eq_LIF, threshold='v>0.8', reset='v=0',method='euler',dt=0.01*b2.second)
    net['global_inhibitory'].tau = tau*b2.second
    net['dis_inhibitory'] = NeuronGroup(
        1, eq_LIF, threshold='v>0.8', reset='v=0',method='euler',dt=0.01*b2.second)
    net['dis_inhibitory'].tau = tau*b2.second

    net['syn_input2hidden'] = Synapses(
        net['input'], net['neu_hidden'], 'w:1', on_pre='v+=w',method='euler',dt=0.01*b2.second)
    net['syn_input2hidden'].connect()
    weights.neuron_count.hidden = args.hidden_size
    net['syn_input2hidden'].w = weights.generate_weights(weights.neuron_count)
    net['syn_hidden2output'] = Synapses(
        net['neu_hidden'], net['neu_output'], on_pre='v+=0.1',method='euler',dt=0.01*b2.second)
    net['syn_hidden2output'].connect()
    net['syn_poisson'] = Synapses(
        net['poisson_input'], net['global_inhibitory'], on_pre='v+=0.3',method='euler',dt=0.01*b2.second)
    net['syn_poisson'].connect()
    net['syn_inhibitory'] = Synapses(
        net['global_inhibitory'], net['neu_hidden'], on_pre='v-=0.2',method='euler',dt=0.01*b2.second)
    net['syn_inhibitory'].connect()
    net['syn_input2disinhibitory'] = Synapses(
        net['input'], net['dis_inhibitory'], on_pre='v+=0.5',method='euler',dt=0.01*b2.second)
    net['syn_input2disinhibitory'].connect()
    net['syn_disinhibitory'] = Synapses(
        net['dis_inhibitory'], net['global_inhibitory'], on_pre='v-=0.5',method='euler',dt=0.01*b2.second)
    net['syn_disinhibitory'].connect()

    net['mon_output'] = b2.SpikeMonitor(net['neu_output'])

    return net


def get_snn(spike_up, spike_dn, net: dict):
    spike_indice = np.concatenate(
        (np.zeros_like(spike_up), np.ones_like(spike_dn)))
    spike_times = np.concatenate((spike_up, spike_dn))*b2.second
    net['input'].set_spikes(spike_indice, spike_times)
    network = Network()
    for key in net.keys():
        network.add(net[key])
    return network


def snn_stage(args):
    data, time, annotation = load_data.load_data(
        args.eeg_path, args.annotation_path, args.channel)
    data, label = load_data.split_data(data, time, annotation)
    if args.epochs == 0:
        epochs = data.shape[0]
    else:
        epochs = args.epochs
    snn_compoent = snn(tau=args.tau,args=args)
    net = Network()
    for key in snn_compoent.keys():
        net.add(snn_compoent[key])
    net.store()
    spike_datas = {}
    spike_snn_datas = {}
    indexs = list(range(data.shape[0]))
    random.shuffle(indexs)
    for epoch in indexs[:epochs]:
        print(epoch)
        spike_data = {}
        spike_snn_data = {}
        signals = filter.get_eeg_band(data[epoch])
        for key in tqdm.tqdm(signals.keys()):
            net.restore()
            spike_up, spike_dn = spike(signals[key])
            spike_data[key] = {'up':list(spike_up),'dn':list(spike_dn)}
            spike_indice = np.concatenate(
                (np.zeros_like(spike_up), np.ones_like(spike_dn)))
            spike_times = np.concatenate((spike_up, spike_dn))*b2.second
            snn_compoent['input'].set_spikes(spike_indice, spike_times)
            net.run(30*b2.second)
            spike_time = snn_compoent['mon_output'].t/b2.second
            spike_snn_data[key] = list(spike_time)
        spike_snn_data['label'] = int(label[epoch])
        spike_datas[epoch] = spike_data
        spike_snn_datas[epoch] = spike_snn_data
        with open(args.spike_save_path+'origional_spike.json','w') as f:
            json.dump(spike_datas,f)
        with open(args.spike_save_path+'snn_spike.json','w') as f:
            json.dump(spike_snn_datas,f)

if __name__ == '__main__':
    args = config_parser().parse_args()
    snn_stage(args)