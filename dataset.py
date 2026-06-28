import torch
from torch.utils.data import Dataset


class FrameDataset(Dataset):
    def __init__(self, observations, rewards):
        self.observations = observations
        self.rewards = rewards

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, i):
        x = torch.from_numpy(self.observations[i]).float() / 255.0
        x = x.permute(2, 0, 1)
        r = torch.tensor(self.rewards[i]).float()
        return x, r