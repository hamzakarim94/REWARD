import torch
from torch import nn
import numpy as np
class MLP(torch.nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.input = nn.Linear(512, 1000)
        self.hidden = nn.Linear(1000, 1000)
        self.hidden2 = nn.Linear(1000, 512)
        self.out = nn.Linear(512, 1)
        self.rel = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x=self.input(x)
        x = self.rel(x)
        x = self.hidden(x)
        x = self.rel(x)
        x = self.hidden2(x)
        x = self.rel(x)
        x = self.out(x)
        x = self.sig(x)
        return x

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def compute_Dt(vals,s=0,l=0):

    if isinstance(vals, np.ndarray):
        vals=vals.tolist()
    st_arr = [0] * len(vals)
    if s > 0:
        del vals[:s]
    if l > 0:
        vals = vals[:-l]
    vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals))
    st =0
    e_alpha = sum(vals) / len(vals)
    for e, anom in enumerate(vals):
        if e == 0:
            st = max((st + anom - e_alpha, 0))
        else:
            st = max((st + anom - e_alpha, 0))
        st_arr[e+s] = st
    return st_arr,e_alpha

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x