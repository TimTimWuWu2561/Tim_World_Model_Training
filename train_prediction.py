import numpy as np
import torch
import torch.nn as nn

from models import InverseActionModel, MemoryModel, PredictionModel, MEMORY_DIM


def train_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    GROUND = 80      # how many real V steps to build memory from before rolling out
    ROLLOUT = 40     # how many steps to predict forward (feeding predictions back in)
    BATCH = 64
    EPOCHS = 20

    # --- build training samples: each sample is one anchor's slice of the run ---
    # a sample needs GROUND real steps + ROLLOUT future steps to compare against.
    # so each sample spans GROUND + ROLLOUT vectors.
    samples = []
    for n in range(1, 11):
        v = np.load(f"vectors/vec_{n}.npy")          # (1000, 256)
        last_start = len(v) - (GROUND + ROLLOUT)
        for start in range(0, last_start + 1):
            samples.append(v[start : start + GROUND + ROLLOUT])

    samples = np.stack(samples)                      # (num_samples, GROUND+ROLLOUT, 256)
    samples = torch.from_numpy(samples).float()
    print("samples shape:", tuple(samples.shape))

    inverse = InverseActionModel().to(device)
    memory  = MemoryModel().to(device)
    predict = PredictionModel().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(inverse.parameters()) + list(memory.parameters()) + list(predict.parameters()),
        lr=1e-3,
    )

    num_samples = samples.shape[0]

    for epoch in range(EPOCHS):
        perm = torch.randperm(num_samples)
        running = 0.0
        count = 0

        for i in range(0, num_samples, BATCH):
            idx = perm[i : i + BATCH]
            batch = samples[idx].to(device)          # (B, GROUND+ROLLOUT, 256)

            # --- phase 1: build memory from the GROUND real steps ---
            ground = batch[:, :GROUND, :]            # (B, GROUND, 256) real V
            v_prev = ground[:, :-1, :]               # V(t-1)
            v_cur  = ground[:, 1:,  :]               # V(t)

            actions = inverse(v_cur, v_prev)         # (B, GROUND-1, ACTION_DIM)
            gru_in = torch.cat([v_cur, actions], dim=-1)

            # run the GROUND steps through the GRU, keep the final hidden state h
            mem_seq, h = memory.gru(gru_in)          # h: (1, B, MEMORY_DIM)
            # last real vector and the memory at the end of grounding
            last_v = ground[:, -1, :]                # V at the anchor (B, 256)
            prev_v = ground[:, -2, :]                # V just before anchor (B, 256)

            # --- phase 2: roll out ROLLOUT predictions, feeding predictions back ---
            preds = []
            cur_v = last_v
            prv_v = prev_v
            for step in range(ROLLOUT):
                # infer action from current and previous vector
                a = inverse(cur_v, prv_v)            # (B, ACTION_DIM)
                step_in = torch.cat([cur_v, a], dim=-1).unsqueeze(1)  # (B, 1, 256+ACTION)
                out, h = memory.gru(step_in, h)      # advance memory one step, carry h
                m = out[:, 0, :]                     # (B, MEMORY_DIM)
                p = predict(m)                       # (B, 256) predicted next V
                preds.append(p)
                # the prediction becomes the next "current"; old current becomes "previous"
                prv_v = cur_v
                cur_v = p

            preds = torch.stack(preds, dim=1)        # (B, ROLLOUT, 256)
            target = batch[:, GROUND:GROUND + ROLLOUT, :]  # real future V (B, ROLLOUT, 256)

            loss = loss_fn(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            count += 1

        print(f"epoch {epoch}  loss {running / count:.4f}")

    torch.save(inverse.state_dict(), "inverse.pth")
    torch.save(memory.state_dict(),  "memory.pth")
    torch.save(predict.state_dict(), "predict.pth")
    print("saved inverse.pth, memory.pth, predict.pth")