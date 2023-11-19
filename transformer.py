import configargparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import tqdm

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--data_path', default='data\spike\snn_spike.json')
    parser.add_argument('--hidden_layer_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--i_save', type=int, default=10)
    parser.add_argument('--output_path', default='./output/')
    return parser


def read_data(args):
    with open(args.data_path, 'r') as f:
        datas = json.load(f)
    signals, labels = [], []
    for key in datas.keys():
        data = datas[key]
        labels.append(data['label'])
        signal = np.zeros((30, 5))
        for i, channel in enumerate(list(data.keys())[:5]):
            for j in data[channel]:
                signal[int(j), i] += 1
        signals.append(signal)
    return signals, labels


class classifier(nn.Module):
    def __init__(self, args):
        super(classifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_layer_dim, nhead=8, dim_feedforward=args.hidden_layer_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=args.num_layers)
        self.embedding = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, args.hidden_layer_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.hidden_layer_dim, 6)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        y = self.classifier(x[:, 0])
        return y


class dataset(Dataset):
    def __init__(self, signals, labels):
        self.signals, self.labels = signals, labels

    def __getitem__(self, index):
        signal = self.signals[index].astype('float32')
        label = self.labels[index]
        return torch.tensor(signal).to(device), torch.tensor(label).to(device)

    def __len__(self):
        return len(self.labels)


def test(net, loader):
    n = 0
    m = 0
    for index, (signal, label) in enumerate(loader):
        y = net(signal)
        if y.argmax() == label:
            m += 1
        n += 1
    return m/n


def main(args):
    signals, labels = read_data(args)
    signal_train, signal_test, label_train, label_test = train_test_split(
        signals, labels, test_size=0.2)
    print('train size:', len(label_train))
    print('test_size:', len(label_test))

    net = classifier(args).to(device)
    train_loader = DataLoader(
        dataset(signal_train, label_train), batch_size=args.batch_size, shuffle=True)
    train_sub_loader = DataLoader(
        dataset(signal_train, label_train), batch_size=1, shuffle=True)
    test_loader = DataLoader(
        dataset(signal_test, label_test), batch_size=1, shuffle=True)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,momentum=0.1,weight_decay=0.01)
    lr_shedular = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs+30)

    train_loss = []
    train_acc = []
    test_acc = []
    epoch_n = []

    for epoch in tqdm.tqdm(range(args.epochs)):
        net.train()
        for index, (signal, label) in enumerate(train_loader):
            y = net(signal)
            loss = loss_fun(y, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(float(loss))
        lr_shedular.step()

        if ((epoch+1) % args.i_save == 0) or epoch == 0:
            epoch_n.append(epoch+1)
            path = args.output_path+'exp'+str(epoch+1)+'/'
            try:
                os.mkdir(path)
            except:
                pass
            net.eval()
            train_acc.append(test(net, train_sub_loader))
            test_acc.append(test(net, test_loader))
            plt.plot(train_loss)
            plt.savefig(path+'loss.png')
            with open(path+'acc.json', 'w') as f:
                json.dump(
                    {'train': train_acc, 'test': test_acc, 'time': epoch_n}, f)
            # torch.save(net,path+'mode.pt')


if __name__ == '__main__':
    args = config_parser().parse_args()
    main(args)
