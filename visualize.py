import numpy as np
import torch
import imageio

from models import Encoder, Decoder, InverseActionModel, MemoryModel, PredictionModel


def _load_decoder(device):
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
    decoder.eval()
    return decoder


def _to_image(tensor):
    # tensor: (3, 96, 96) float 0..1  ->  (96, 96, 3) uint8
    img = tensor.permute(1, 2, 0)
    img = (img * 255).clamp(0, 255)
    return img.byte().cpu().numpy()


def watch_reconstruction():
    """Encode then decode each frame of run 1, save as a video to eyeball the autoencoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("encoder.pth", map_location=device))
    encoder.eval()
    decoder = _load_decoder(device)

    run = np.load("runs/run_1.npz")
    frames = run["observations"]                     # (1000, 96, 96, 3) uint8

    out_frames = []
    with torch.no_grad():
        for i in range(len(frames)):
            x = torch.from_numpy(frames[i]).float() / 255.0
            x = x.permute(2, 0, 1).unsqueeze(0).to(device)   # (1, 3, 96, 96)

            v = encoder(x)
            recon = decoder(v).squeeze(0)                     # (3, 96, 96)

            original = torch.from_numpy(frames[i]).float() / 255.0
            original = original.permute(2, 0, 1)             # (3, 96, 96)

            # put original (left) and reconstruction (right) side by side
            pair = torch.cat([original.to(device), recon], dim=2)  # (3, 96, 192)
            out_frames.append(_to_image(pair))

    imageio.mimsave("reconstruction.mp4", out_frames, fps=30)
    print("saved reconstruction.mp4  (left = original, right = reconstruction)")


def watch_prediction():
    """Run run 1 through the prediction chain, decode predictions, save predicted-vs-actual video."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    decoder = _load_decoder(device)
    inverse = InverseActionModel().to(device)
    memory  = MemoryModel().to(device)
    predict = PredictionModel().to(device)
    inverse.load_state_dict(torch.load("inverse.pth", map_location=device))
    memory.load_state_dict(torch.load("memory.pth", map_location=device))
    predict.load_state_dict(torch.load("predict.pth", map_location=device))
    inverse.eval()
    memory.eval()
    predict.eval()

    v = np.load("vectors/vec_1.npy")                 # (1000, 256)
    v = torch.from_numpy(v).float().unsqueeze(0).to(device)   # (1, 1000, 256)

    with torch.no_grad():
        v_prev = v[:, :-2, :]                        # V(t-1)
        v_cur  = v[:, 1:-1, :]                       # V(t)
        v_next = v[:, 2:,  :]                        # V(t+1) actual next

        actions = inverse(v_cur, v_prev)
        mem_seq = memory(v_cur, actions)
        pred    = predict(mem_seq)                   # (1, len, 256) predicted next

        # decode predicted next vectors and actual next vectors into images
        pred_imgs = decoder(pred.squeeze(0))         # (len, 3, 96, 96)
        real_imgs = decoder(v_next.squeeze(0))       # (len, 3, 96, 96)

    out_frames = []
    for i in range(pred_imgs.shape[0]):
        pair = torch.cat([real_imgs[i], pred_imgs[i]], dim=2)  # (3, 96, 192)
        out_frames.append(_to_image(pair))

    imageio.mimsave("prediction.mp4", out_frames, fps=30)
    print("saved prediction.mp4  (left = actual next frame, right = predicted next frame)")