from torch.utils.data import Dataset
import numpy as np
# from math import abs


class RLdataset(Dataset):
    def __init__(self,threshold,thresholdradians) -> None:
        super(RLdataset,self).__init__()
        self.currentstate = np.load("../currentstate.npy")
        self.action = np.load("../action.npy")
        # self.reward = np.load("../reward.npy")
        self.nextstate = np.load("../nextstate.npy")
        self.threshold = threshold
        self.thresholdradians =  thresholdradians
    def __len__(self):
        return len(self.action)
    
    def _getrealreward(self,state):
        x,x_dot,theta,theta_dot = state
        r1 = (self.threshold - abs(x))/self.threshold - 0.8
        r2 = (self.thresholdradians - abs(theta))/self.thresholdradians - 0.5
        return r1 + r2
    def __getitem__(self, index):
        ns = self.nextstate[index]
        reward = self._getrealreward(ns)
        return self.currentstate[index],self.action[index],reward,ns

       