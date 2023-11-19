import torch
import torchvision
import torch.nn as nn
import utils.filter
import SNN

class sleep_classifier(nn.Module):
    def __init__(self):
        super(sleep_classifier,self).__init__()
        self.snn_compoent = SNN.snn(tau=0.1)
        