import numpy as np
import torch
import os

from models import Encoder


def precompute_vectors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    # load the trained encoder
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    encoder.eval()
    print("loaded encoder.pth")

    os.makedirs("vectors", exist_ok=True)

    with torch.no_grad():
        for n in range(1, 11):
            run = np.load(f"runs/run_{n}.npz")
            frames = run["observations"]                 # (1000, 96, 96, 3) uint8

            x = torch.from_numpy(frames).float() / 255.0 # (1000, 96, 96, 3)
            x = x.permute(0, 3, 1, 2)                     # (1000, 3, 96, 96)
            x = x.to(device)

            v = encoder(x)                               # (1000, LATENT_DIM)
            v = v.cpu().numpy()

            np.save(f"vectors/vec_{n}.npy", v)
            print(f"saved vectors/vec_{n}.npy  shape {v.shape}")