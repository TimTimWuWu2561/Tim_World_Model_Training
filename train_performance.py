import numpy as np
import torch
import torch.nn as nn

from models import PerformanceEvaluationModel, HISTORY


def train_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    BATCH = 64
    EPOCHS = 40

    # --- build (V-history, reward) pairs ---
    # input  = V(t-HISTORY+1) .. V(t)   (the last HISTORY frames ending at t)
    # target = reward at t
    inputs = []
    targets = []
    for n in range(1, 11):
        v = np.load(f"vectors/vec_{n}.npy")          # (1000, 256)
        run = np.load(f"runs/run_{n}.npz")
        rewards = run["rewards"]                      # (1000,)

        # t must be at least HISTORY-1 so a full window exists behind it
        for t in range(HISTORY - 1, len(v)):
            inputs.append(v[t - HISTORY + 1 : t + 1])   # (HISTORY, 256)
            targets.append(rewards[t])

    inputs = np.stack(inputs)                        # (num, HISTORY, 256)
    targets = np.array(targets, dtype=np.float32)    # (num,)
    inputs = torch.from_numpy(inputs).float()
    targets = torch.from_numpy(targets).float().unsqueeze(1)   # (num, 1)
    print("inputs shape:", tuple(inputs.shape), " targets shape:", tuple(targets.shape))

    model = PerformanceEvaluationModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    num = inputs.shape[0]
    for epoch in range(EPOCHS):
        perm = torch.randperm(num)
        running = 0.0
        count = 0
        for i in range(0, num, BATCH):
            idx = perm[i : i + BATCH]
            x = inputs[idx].to(device)
            y = targets[idx].to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            count += 1

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        print(f"epoch {epoch}  loss {running / count:.4f}  lr {lr_now:.1e}")

    torch.save(model.state_dict(), "performance.pth")
    print("saved performance.pth")