from offlinerldataset.offlinerl import RLdataset

if __name__ == "__main__":
    inst = RLdataset(0.8,0.1)
    from torch.utils.data import DataLoader
    for state,action,reward,nextstate in DataLoader(inst):
        print(state.shape)
        print(action.shape)
        print(reward.shape)
        print(nextstate.shape)
        exit()