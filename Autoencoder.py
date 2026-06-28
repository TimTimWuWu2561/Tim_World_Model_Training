import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------------
# 1. DATA
# Wraps a numpy array of frames so the DataLoader can hand them out in
# batches. Your real frames go in place of the random array below.
# -------------------------------------------------------------------
class FrameDataset(Dataset):
    def __init__(self, frames):
        # frames: numpy array, shape (N, 96, 96, 3), dtype uint8
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        frame = self.frames[i]                       # (96, 96, 3) uint8
        x = torch.from_numpy(frame).float() / 255.0  # -> float, range 0..1
        x = x.permute(2, 0, 1)                        # -> (3, 96, 96)
        return x

# Fake data so this runs out of the box. Replace with your real frames.
fake_frames = np.random.randint(0, 256, size=(500, 96, 96, 3), dtype=np.uint8)
dataset = FrameDataset(fake_frames)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------------------------------------------------
# 2. THE TWO NETWORKS
# -------------------------------------------------------------------
LATENT_DIM = 256   # the bottleneck vector size. This is the dial you tune.

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # each conv (stride 2) halves the spatial size: 96 -> 48 -> 24 -> 12
        self.net = nn.Sequential(
            nn.Conv2d(3,   32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32,  64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(),
        )
        self.to_vector = nn.Linear(128 * 12 * 12, LATENT_DIM)

    def forward(self, x):
        x = self.net(x)                 # (batch, 128, 12, 12)
        x = x.flatten(start_dim=1)      # (batch, 128*12*12)
        v = self.to_vector(x)           # (batch, LATENT_DIM)
        return v

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.from_vector = nn.Linear(LATENT_DIM, 128 * 12 * 12)
        # each transposed conv (stride 2) doubles the size: 12 -> 24 -> 48 -> 96
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64,  32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32,   3, kernel_size=4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, v):
        x = self.from_vector(v)         # (batch, 128*12*12)
        x = x.view(-1, 128, 12, 12)     # back to image-shaped
        x = self.net(x)                 # (batch, 3, 96, 96), values 0..1
        return x

encoder = Encoder()
decoder = Decoder()

# -------------------------------------------------------------------
# 3. LOSS + OPTIMIZER
# One optimizer gets BOTH networks' parameters, so both get trained.
# -------------------------------------------------------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=1e-3,
)

# -------------------------------------------------------------------
# 4. TRAINING LOOP
# -------------------------------------------------------------------
for epoch in range(5):
    for batch in loader:
        v = encoder(batch)              # image  -> vector
        recon = decoder(v)              # vector -> reconstructed image
        loss = loss_fn(recon, batch)    # how wrong the reconstruction is

        optimizer.zero_grad()           # clear last step's gradients
        loss.backward()                 # compute gradients for both nets
        optimizer.step()                # nudge all weights

    print(f"epoch {epoch}  loss {loss.item():.4f}")