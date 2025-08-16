import torch
from src.models import MLP

m=MLP(hidden=256,dropout=0.1)
x=torch.randn(8,1,28,28)
y=m(x)
print('output shape:', tuple(y.shape))

num_params = sum(p.numel() for p in m.parameters())
print("num params:", num_params)
