import torch
import torch.nn as nn

class MLP(nn.Module): #MULTI-LAYER PERCEPTRON
    def __init__(self, hidden=256, dropout=0.1):
        super().__init__()
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(28*28, hidden)
        self.act=nn.ReLU()
        self.drop=nn.Dropout(dropout)
        self.fc2=nn.Linear(hidden,10)

    def forward(self, x):
        x=self.flatten(x)
        x=self.fc1(x)
        x=self.act(x)
        x=self.drop(x)
        x=self.fc2(x)
        return x