import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FrameDataset
from models import Encoder, Decoder

EPOCHS = 30

def train_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    # load all 10 runs and combine them
    obs_list = []
    rew_list = []
    for n in range(1, 11):
        run = np.load(f"runs/run_{n}.npz")
        obs_list.append(run["observations"])
        rew_list.append(run["rewards"])

    observations = np.concatenate(obs_list, axis=0)
    rewards = np.concatenate(rew_list, axis=0)
    print("total frames:", len(observations))

    dataset = FrameDataset(observations, rewards)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3,
    )

    for epoch in range(EPOCHS):
        for images, rewards_batch in loader:
            images = images.to(device)
            v = encoder(images)
            recon = decoder(v)
            loss = loss_fn(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch}  loss {loss.item():.4f}")

    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")
    print("saved encoder.pth and decoder.pth")