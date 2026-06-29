# World Model Behavior-Learning Agent — Project README

This document is a complete handoff. If you are an AI assistant reading this for
the first time, it tells you exactly what this project is, what has been built,
how each piece works, the order things run in, and what comes next. The user is
**new to PyTorch and is learning as they build** — explanations should be clear,
concrete, and should not assume prior deep-learning knowledge. The user strongly
prefers the straightforward approach over clever tricks, and dislikes being given
optional "tricks" presented as if they were required decisions. Explain things
plainly, and when showing code, say what each unfamiliar line does to the data.

---

## 1. What this project is

The goal is to build a **world model agent** that learns to play the game
**CarRacing-v3** (from Gymnasium). The architecture follows a design where the
agent learns, in stages:

1. to **compress** what it sees (raw game frames) into small vectors,
2. to **predict** what will happen next (so it can "dream" / imagine gameplay),
3. to **evaluate** how good a situation is (using the game's reward),
4. (not built yet) to **decide** what action to take.

The big idea: instead of learning to act directly from pixels, the agent first
builds an internal model of the game — vision, memory, prediction, and value —
and will eventually use that internal model to choose actions, including by
"dreaming" (training on imagined gameplay rather than only real gameplay).

The raw data is **RGB frames of shape (96, 96, 3)** plus a **reward** at every
timestep. There is **no recorded action** — actions were intentionally not saved,
which is why an "inverse action" model exists to infer actions from how the vision
changes.

---

## 2. Current progress

**Built and working (verified):**

- **Abstraction model** (autoencoder): compresses a frame into a 256-number vector
  and reconstructs it. Verified by watching reconstruction videos.
- **Prediction module** (inverse action + memory + prediction): can "dream" —
  given some real frames to start, it predicts forward on its own predictions.
  Verified by watching dream videos; stays coherent well past its training horizon.
- **Performance evaluation model**: given recent vision, predicts the reward
  (how good the situation is). Trained and working (loss ~0.028, reasonable).

**Not built yet:**

- **Action model** (the decision-maker) — the next major piece.
- **Training methods** from the design diagrams: "dream training" and
  "practice training", which use the above modules to train the action model.

The user considers the current models "good enough for now" and is comfortable
refining them later. They are not chasing perfect numbers; they want reasonable,
working pieces and forward progress.

---

## 3. The data

- **Raw runs**: stored in `runs/run_1.npz` … `runs/run_10.npz`. Ten recorded
  playthroughs, each **1000 frames**. Each `.npz` holds two arrays:
  - `observations`: shape (1000, 96, 96, 3), dtype uint8 (raw RGB frames)
  - `rewards`: shape (1000,), float (reward at each timestep)
- **Reward structure** (important): mostly a constant **−0.1** per frame (a time
  penalty), with frequent **~3.3** spikes when the car reaches new track tiles.
  So it's roughly a two-level signal: −0.1 (normal) vs ~3.3 (made progress).
- **Precomputed vision vectors**: `vectors/vec_1.npy` … `vectors/vec_10.npy`,
  each shape (1000, 256). These are the encoder's output for every frame, saved
  so the later models train on vectors instead of re-encoding images every time.
  `vec_N[t]` corresponds to `run_N`'s frame `t` (aligned by index).

Raw data and generated files (`runs/`, `vectors/`, `*.pth`, `*.mp4`) are **not in
git** (see `.gitignore`); only code is versioned. To run on another machine, the
`runs/` folder must be copied over; everything else regenerates by running the code.

---

## 4. How each model works

All models live in `models.py`. Key sizes (named constants at the top of that file):
- `LATENT_DIM = 256` — the vision vector size (V).
- `ACTION_DIM = 3` — inferred action size (matches steering / gas / brake).
- `MEMORY_DIM = 768` — the memory state size (M).
- `HISTORY = 50` — how many past frames the performance model looks at.

### 4.1 Encoder + Decoder (the abstraction model)

An **autoencoder**: two networks trained together as one.

- **Encoder**: takes a frame (3, 96, 96) and squeezes it to a 256-number vector.
  It uses three **convolutional layers** (each with `stride=2`, which halves the
  image size: 96 → 48 → 24 → 12) that grow the channel count (3 → 32 → 64 → 128),
  then flattens (128 × 12 × 12 = 18,432 numbers) and a **Linear layer** compresses
  that to 256. The 256-vector is the "bottleneck" — the compressed representation.
- **Decoder**: the mirror — takes the 256-vector, expands it with a Linear layer,
  reshapes to (128, 12, 12), then three **transposed-convolution layers**
  (`stride=2`, which *doubles* the size: 12 → 24 → 48 → 96) rebuild the image. A
  final **Sigmoid** squashes outputs to 0–1 to match the input's pixel range.
- **Training**: feed a frame in, get a reconstruction out, compare with MSE loss,
  backprop through both networks at once. The bottleneck forces the encoder to
  pack the important information into 256 numbers.

### 4.2 Inverse Action Model

Plain feed-forward network. Takes **V(t) and V(t-1)** (current and previous vision
vectors), glues them together (`torch.cat`, 512 numbers), and outputs a 3-number
**inferred action**. Because real actions were never recorded, this model recovers
"what action happened between these two frames" from how the vision changed. It is
**not trained on its own** — it has no direct target. It is trained jointly with
the memory and prediction models via the prediction loss (it learns to produce
whatever "action" makes downstream prediction accurate).

### 4.3 Memory Model (a GRU)

A **GRU** is a *recurrent* layer — it processes a sequence one step at a time,
carrying a memory state forward. At each step it takes the current input plus the
memory-so-far and produces an updated memory. (GRU = "Gated Recurrent Unit"; the
"gates" are internal learned controls that decide how much old memory to keep vs.
overwrite. PyTorch's `nn.GRU` handles this; you don't implement the gates.)

The memory model's input at each step is **V(t) glued with the inferred action**
(256 + 3 = 259 numbers). Its memory state is `MEMORY_DIM = 768`. It outputs the
updated memory M(t) at each step. The memory accumulates the history of vision +
actions over time, and is what the prediction model uses to predict the future.

### 4.4 Prediction Model

Plain feed-forward network. Takes the memory **M(t)** and outputs a predicted
**next vision vector P(t)** (256 numbers, same size as a real V). During training,
P(t) is compared against the actual next vector V(t+1) — that comparison is the
prediction loss. ("Compare P_t and V_t+1" in the design diagram.)

### 4.5 How the prediction module is trained (multi-step rollout)

This is the most important and least obvious part. The three models above
(inverse, memory, prediction) are trained **together**, with one loss, using a
**multi-step rollout** so the model learns to "dream" (predict from its own
predictions) rather than only predict one step from real data.

Settings (in `train_prediction.py`): `GROUND = 80`, `ROLLOUT = 40`.

For each training sample:
1. **Grounding phase**: feed 80 real V vectors through inverse → GRU to build up
   the memory state. (This is done efficiently by passing the whole sequence to
   the GRU at once.)
2. **Rollout phase**: predict 40 steps forward, but **feed each prediction back in
   as the next input** (stepping the GRU one timestep at a time, carrying the
   memory forward). So after step 1 of the rollout, everything runs on *predicted*
   vectors — this is the "dream" condition where errors compound.
3. Compare the 40 predictions against the 40 real future vectors (MSE), and
   backprop through the whole 40-step chain.

This multi-step training is what makes the model able to dream coherently for many
steps instead of falling apart immediately. A learning-rate scheduler (`StepLR`)
is used; final loss reached ~0.0157.

### 4.6 Performance Evaluation Model (a GRU)

Judges **how good a situation is**, as a single number, from recent vision. It is
*not* evaluating an action — it evaluates the situation the recent frames show.

- **Input**: the last `HISTORY = 50` vision vectors, as a sequence.
- **Architecture**: a **GRU** reads the 50 vectors in order and summarizes them
  into one memory vector; a Linear layer maps that to one value.
- **Target**: the **reward at the current timestep** (immediate reward, not
  discounted future reward — kept simple for now).
- **Why a GRU instead of flattening**: originally this used flattening (glue all
  50 vectors into one big input). The GRU version scales better to long windows
  (its size doesn't grow with window length) and reads frames in order. It trained
  to a much lower loss.
- It will be used later during dream training to evaluate dreamed situations
  (situations the model generates, that it has never literally seen).

> Note on possible overfitting: with only 10 runs, a powerful model can reach very
> low loss by memorizing rather than generalizing. The proper check (held out runs
> for testing) was intentionally skipped for now to keep things simple. If this
> model later judges dreamed situations poorly, revisit with a train/test split.

---

## 5. The code files and the workflow

The program is run through a **menu** in `main.py`. Run `python main.py`, and it
prints a numbered menu; type a number to run that stage, and it returns to the menu
afterward (so you can run several stages in one session). This is designed to be
extended: each stage is a function in its own file, imported into `main.py` and
added to the `OPERATIONS` dictionary with one line.

### Files

- `main.py` — the menu dispatcher.
- `models.py` — all network class definitions (Encoder, Decoder,
  InverseActionModel, MemoryModel, PredictionModel, PerformanceEvaluationModel)
  and the size constants.
- `dataset.py` — `FrameDataset` (used when training the autoencoder on raw frames).
- `train_encoder.py` — trains the autoencoder, saves `encoder.pth` and `decoder.pth`.
- `precompute_vectors.py` — loads the trained encoder, runs every frame through it,
  saves V vectors to `vectors/`.
- `train_prediction.py` — multi-step rollout training of inverse + memory +
  prediction; saves `inverse.pth`, `memory.pth`, `predict.pth`.
- `train_performance.py` — trains the performance evaluation model; saves
  `performance.pth`.
- `visualize.py` — three "watch" functions (reconstruction, single-step prediction,
  dream) that decode vectors back into images and save videos.
- `inspect_rewards.py` — standalone helper to print reward statistics (run directly).
- (game recording script — kept separately; plays CarRacing with keyboard input
  and saves a run as `runs/run_N.npz`. Needs a display, so cannot run on a headless
  server.)

### Menu options (in `main.py`)

1. Train abstraction model (autoencoder) → produces `encoder.pth`, `decoder.pth`
2. Precompute V vectors from trained encoder → produces `vectors/`
3. Train prediction module (inverse + memory + prediction) → produces the three .pth
4. Watch abstraction reconstructions → `reconstruction.mp4` (original vs. rebuilt)
5. Watch predictions vs actual (single-step) → `prediction.mp4`
6. Watch dream (ground on real, then predict forward) → `dream.mp4` (actual vs. dream)
7. Train performance evaluation model → produces `performance.pth`

### Order to run (first time, from raw data)

Because each stage consumes the previous stage's output files, run in order:

1. Option **1** (train encoder) — needs `runs/`.
2. Option **2** (precompute vectors) — needs `encoder.pth`.
3. Option **3** (train prediction) — needs `vectors/`.
4. Option **7** (train performance) — needs `vectors/` and `runs/`.

Visualization options 4/5/6 can be run after their respective models exist.
Once the .pth weight files exist, any single stage can be re-run on its own without
redoing earlier ones (e.g. retrain just the prediction module).

### Environment / libraries

- Python with PyTorch (GPU build via CUDA; user has an NVIDIA RTX 2060 at home).
- Other libraries: `numpy`, `imageio` + `imageio[ffmpeg]` (for mp4 videos),
  `gymnasium[box2d]` and `pygame` (only needed for recording new gameplay, not for
  training). Install PyTorch from the official selector at pytorch.org with the
  CUDA index URL for GPU support.

---

## 6. What's next (future work)

**Immediate next major piece: the Action Model.** This is the decision-maker — the
part that actually chooses what to do. It will use the modules already built
(vision, memory, prediction, performance evaluation). Design decisions to settle
with the user before coding: what it takes as input (likely memory and/or vision),
what it outputs (an action — 3 numbers like steering/gas/brake), and how it is
trained.

**Training methods (from the design diagrams):**
- **Dream training**: train the action model inside *imagined* gameplay produced by
  the prediction module — the agent dreams a situation, the performance model
  evaluates how good it is, and the action model learns to pick actions that lead to
  good (high-value) dreamed situations. This is the payoff of building the world
  model: training without needing constant real gameplay.
- **Practice training**: train using real gameplay, refining both prediction and
  action models.

**Possible refinements to existing models (optional, later):**
- Longer prediction rollouts (ramp-up) so dreams stay coherent for more steps.
- Train/test split on the performance model to check for overfitting.
- More and more varied recorded runs for robustness.

**The design diagrams** (the user has hand-drawn architecture diagrams) describe
the full intended system: an Abstraction Model, a Performance Evaluation Model, and
three training methods (Dataset / Dream / Practice) that combine vision, memory,
inverse action, prediction, and action models. The pieces built so far implement
the abstraction, the prediction module, and the performance evaluation; the action
model and the dream/practice training loops are the remaining core work.

---

## 7. How to work with this user

- They are learning PyTorch from scratch and want to **understand**, not just run
  code. Explain what unfamiliar lines do to the actual data (shapes, numbers).
- Prefer the **simple, direct approach**. Do not introduce optional "tricks" as if
  they were mandatory choices. If a clever optimization exists, mention it only as
  a clearly-labeled optional bonus, never as a fork in the road.
- When they propose a simpler way to do something, take it seriously — they are
  often right, and over-complication has caused real frustration before.
- They are comfortable with "good enough for now, fix later." Don't insist on
  rigor (like train/test splits) when they've chosen to move fast — flag it once,
  then respect their call.
- Keep responses focused; avoid burying the answer in caveats.