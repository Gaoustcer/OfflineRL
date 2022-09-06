import gym

import numpy as np
from torch.distributions import Categorical
def _connect():
    env = gym.make("CartPole-v0")
    currentstate = []
    action = []
    reward = []
    nextstate = []
    done = []
    EPOSIDETIME = 1024
    for epoch in range(EPOSIDETIME):
        state = env.reset()
        d = False
        while d == False:
            a = env.action_space.sample()
            ns,r,d,_ = env.step(a)
            currentstate.append(state)
            action.append(a)
            reward.append(r)
            nextstate.append(ns)
            done.append(d)
    np.save("currentstate.npy",np.array(currentstate))
    np.save('action.npy',np.array(action))
    np.save('nextstate.npy',np.array(nextstate))
    np.save('done.npy',np.array(done))
if __name__ == "__main__":
    _connect()