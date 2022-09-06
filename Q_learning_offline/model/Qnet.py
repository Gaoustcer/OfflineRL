import torch.nn as nn
import numpy as np
import torch

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(4,2),
            nn.ReLU(),
            nn.Linear(2,2)
            # nn.Softmax()
        )
    
    def forward(self,x):
        if isinstance(x,np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x,torch.Tensor):
            x = x.cuda()
            return self.linear(x)
        else:
            raise TypeError
            