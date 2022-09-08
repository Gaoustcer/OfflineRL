import torch

from model.Qnet import Net
import numpy as np

import os
import gym
ENV = 'CartPole-v0'
from torch.distributions import Categorical
from offlinerldataset.offlinerl import RLdataset
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

class TrainingAgent(object):
    def __init__(self) -> None:
        self.device = torch.device('cuda:0')
        self.Qnet = Net().to(self.device)
        self.TargetQnet = Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.Qnet.parameters(), lr = 4e-4)
        self.epoch = 256
        self.validenv = gym.make(ENV)
        self.vaildationtime = 16
        self.lossindex = 1
        self.mydataset = RLdataset(self.validenv.x_threshold,self.validenv.theta_threshold_radians)
        self.mydataloader = DataLoader(self.mydataset,batch_size=32)

    def _trainanepoch(self):
        from tqdm import tqdm
        for currentstate,action,reward,nextstate in (self.mydataloader):
            values = self.Qnet(currentstate)
            # (s,r,a,s')
            # Q(s,a) - \max_{a'} Q(s',a')+r
            self.optimizer.zero_grad()
            action = action.cuda()
            currentQvalues = torch.gather(values,-1,action.unsqueeze(-1)).squeeze()
            nextQvalyes = torch.max(self.TargetQnet(nextstate),-1)[0].detach() + reward.to(torch.float32).cuda()
            TDlossfunction = torch.nn.MSELoss()
            TDloss = TDlossfunction(currentQvalues,nextQvalyes)
            # self.optimizer.zero_grad()
            TDloss.backward()
            self.optimizer.step()

    def _trainanepochprob(self):
        from tqdm import tqdm
        for currentstate,action,reward,nextstate in (self.mydataloader):
            values = self.Qnet(currentstate)
            # (s,r,a,s')
            # Q(s,a) - \max_{a'} Q(s',a')+r
            self.optimizer.zero_grad()
            action = action.cuda()
            currentQvalues = torch.gather(values,-1,action.unsqueeze(-1)).squeeze()
            nextQvalues = torch.max(self.TargetQnet(nextstate),-1)[0].detach()
            probnextQ = softmax(nextQvalues)
            nextQvalues = torch.einsum('ij,ij->i',nextQvalues,probnextQ)
            TDlossfunction = torch.nn.MSELoss()
            TDloss = TDlossfunction(currentQvalues,nextQvalues)
            # self.optimizer.zero_grad()
            self.writer.add_scalar('TDloss',TDloss,self.lossindex)
            self.lossindex += 1
            TDloss.backward()
            self.optimizer.step()



    def vaildate(self):
        reward = 0
        for _ in range(self.vaildationtime):
            done = False
            state = self.validenv.reset()
            while done == False:
                # action = torch.argmax(self.Qnet(state)).item()
                prob = softmax(self.Qnet(state))
                action = Categorical(prob).sample().cpu().item()
                ns,r,done,_ = self.validenv.step(action)
                state = ns
                reward += r
        return reward/self.vaildationtime

    def _random(self):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter("../log/randombaseline")
        from tqdm import tqdm
        for epoch in tqdm(range(self.epoch)):
            reward = self.vaildate()
            writer.add_scalar('reward',reward,epoch)

    def _train(self):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter('../log/offlineQlearning1024')
        from tqdm import tqdm
        for epoch in tqdm(range(self.epoch)):
            self._trainanepoch()
            if epoch % 8 == 0:
                self.TargetQnet.load_state_dict(self.Qnet.state_dict())
            reward = self.vaildate()
            self.writer.add_scalar('reward',reward,epoch)


if __name__ == "__main__":
    agent = TrainingAgent()
    agent._train()    
    # agent._random()    
