import torch
import torch.nn as nn

LATENT_DIM = 256    # vision vector size (V)
ACTION_DIM = 3      # inferred action size (matches steering/gas/brake)
MEMORY_DIM = 768    # memory state size (M)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # each size: 96 -> 48 -> 24 -> 12
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


class InverseActionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input is V(t) and V(t-1) glued together -> 2 * LATENT_DIM
        self.net = nn.Sequential(
            nn.Linear(2 * LATENT_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
        )

    def forward(self, v_t, v_prev):
        # glue the two vectors end-to-end along the feature dimension
        x = torch.cat([v_t, v_prev], dim=-1)   # (..., 2 * LATENT_DIM)
        action = self.net(x)                   # (..., ACTION_DIM)
        return action


class MemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input at each step: V(t) glued with action I(t) -> LATENT_DIM + ACTION_DIM
        self.gru = nn.GRU(
            input_size=LATENT_DIM + ACTION_DIM,
            hidden_size=MEMORY_DIM,
            batch_first=True,   # input shape is (batch, sequence, features)
        )

    def forward(self, v_seq, action_seq):
        # v_seq:      (batch, seq_len, LATENT_DIM)
        # action_seq: (batch, seq_len, ACTION_DIM)
        x = torch.cat([v_seq, action_seq], dim=-1)   # (batch, seq_len, LATENT_DIM + ACTION_DIM)
        memory_seq, _ = self.gru(x)                  # (batch, seq_len, MEMORY_DIM)
        return memory_seq


class PredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input is memory M(t); output is a predicted next vision vector (size LATENT_DIM)
        self.net = nn.Sequential(
            nn.Linear(MEMORY_DIM, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, LATENT_DIM),
        )

    def forward(self, memory):
        pred = self.net(memory)   # (..., LATENT_DIM)
        return pred


HISTORY = 50        # how many past V vectors the performance model looks at
PERF_HIDDEN = 256   # GRU memory size for the performance model


class PerformanceEvaluationModel(nn.Module):
    def __init__(self):
        super().__init__()
        # read the HISTORY frames as a sequence; the GRU summarizes them
        self.gru = nn.GRU(
            input_size=LATENT_DIM,
            hidden_size=PERF_HIDDEN,
            batch_first=True,    # input shape (batch, HISTORY, LATENT_DIM)
        )
        self.to_value = nn.Linear(PERF_HIDDEN, 1)

    def forward(self, v_history):
        # v_history: (batch, HISTORY, LATENT_DIM)
        out, h = self.gru(v_history)     # h: (1, batch, PERF_HIDDEN) final memory
        summary = h[0]                   # (batch, PERF_HIDDEN)
        value = self.to_value(summary)   # (batch, 1)
        return value