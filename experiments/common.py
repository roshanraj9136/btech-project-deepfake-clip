"""Shared helpers: in-memory datasets, CLIP preprocessing, feature extraction, linear probes.

The side-network model itself is imported unchanged from ../dfd_fcg_cifar10_experiment.py,
so every number here is for the same architecture as the original training run.
"""
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")        # cached numpy copies of the datasets (not in git)
RESULTS_DIR = os.path.join(REPO, "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, REPO)
from dfd_fcg_cifar10_experiment import CLIPWithSideNetworkCIFAR  # noqa: E402  (original architecture)

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
TAP_INDICES = [2, 5, 8, 11]  # 0-indexed CLIP blocks the side network reads = layers 3, 6, 9, 12

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAKE_CLASSES = ['FAKE', 'REAL']  # label order used by the HF dataset dragonintelligence/CIFAKE-image-dataset

HF_SOURCES = {
    "cifar10": ("uoft-cs/cifar10", "img"),
    "cifake": ("dragonintelligence/CIFAKE-image-dataset", "image"),
}


def keep_awake(on=True):
    """Stop Windows from sleeping during long GPU runs (same trick as the original script)."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 if on else 0x80000000)
    except Exception:
        pass


def load_arrays(name):
    """Return uint8 arrays (x_train NHWC, y_train, x_test, y_test), cached as .npz after the first load."""
    cache = os.path.join(DATA_DIR, f"{name}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]

    from datasets import load_dataset
    repo, col = HF_SOURCES[name]
    raw = load_dataset(repo)

    def to_np(split):
        ds = raw[split]
        xs, ys = [], []
        for i in range(0, len(ds), 2000):
            chunk = ds[i:i + 2000]
            xs.append(np.stack([np.asarray(im.convert("RGB")) for im in chunk[col]]))
            ys.append(np.asarray(chunk["label"]))
        x, y = np.concatenate(xs), np.concatenate(ys)
        assert x.shape[1:] == (32, 32, 3), x.shape
        return x, y.astype(np.int64)

    x_tr, y_tr = to_np("train")
    x_te, y_te = to_np("test")
    np.savez(cache, x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
    return x_tr, y_tr, x_te, y_te


def to_float(xb_uint8, device):
    """uint8 NHWC batch -> float NCHW in [0, 1] (identical to transforms.ToTensor())."""
    return torch.from_numpy(xb_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)


def clip_preprocess(x):
    """32x32 -> 224x224 bicubic + CLIP normalisation, exactly as CLIPWithSideNetworkCIFAR.forward does it."""
    x = TF.resize(x, [224, 224], interpolation=transforms.InterpolationMode.BICUBIC)
    return TF.normalize(x, mean=CLIP_MEAN, std=CLIP_STD)


@torch.no_grad()
def extract_clip_features(vit, x_uint8, device, batch_size=256):
    """Frozen CLIP features for linear probes.

    Returns dict:
      'final'  -> [N, 768] final CLS after the last LayerNorm (the exact vector the side-net classifier gets as clip_cls)
      'L3','L6','L9','L12' -> [N, 768] raw CLS token after that transformer block
    """
    vit.eval()
    out = {k: [] for k in ["final", "L3", "L6", "L9", "L12"]}
    for i in range(0, len(x_uint8), batch_size):
        x = clip_preprocess(to_float(x_uint8[i:i + batch_size], device))
        with torch.amp.autocast("cuda"):
            t = vit.patch_embed(x)
            t = vit._pos_embed(t)
            t = vit.patch_drop(t)
            t = vit.norm_pre(t)
            for bi, block in enumerate(vit.blocks):
                t = block(t)
                if bi in TAP_INDICES:
                    out[f"L{bi + 1}"].append(t[:, 0].float().cpu())
            t = vit.norm(t)
            out["final"].append(vit.forward_head(t, pre_logits=True).float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def linear_probe(x_tr, y_tr, x_te, y_te, n_classes, device, lambdas=(1e-5, 1e-4, 1e-3, 1e-2), seed=0):
    """Multinomial logistic regression on frozen features (a 'linear probe').

    L2 strength picked on a 10% validation split of the training set, then refit on all training data.
    Full-batch L-BFGS on the GPU. Returns dict with test accuracy, chosen lambda and test probabilities.
    """
    g = torch.Generator().manual_seed(seed)
    mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True) + 1e-6
    xtr = ((x_tr - mean) / std).to(device)
    xte = ((x_te - mean) / std).to(device)
    ytr = torch.as_tensor(y_tr).to(device)
    yte = torch.as_tensor(y_te).to(device)

    perm = torch.randperm(len(xtr), generator=g).to(device)
    n_val = len(xtr) // 10
    val_idx, fit_idx = perm[:n_val], perm[n_val:]

    def fit(x, y, lam):
        lin = torch.nn.Linear(x.shape[1], n_classes).to(device)
        opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=300, history_size=20,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(lin(x), y) + lam * lin.weight.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        return lin

    val_scores = {}
    for lam in lambdas:
        lin = fit(xtr[fit_idx], ytr[fit_idx], lam)
        with torch.no_grad():
            val_scores[lam] = (lin(xtr[val_idx]).argmax(1) == ytr[val_idx]).float().mean().item()
    best = max(val_scores, key=val_scores.get)
    lin = fit(xtr, ytr, best)
    with torch.no_grad():
        probs = lin(xte).softmax(1).cpu()
    acc = (probs.argmax(1) == yte.cpu()).float().mean().item()
    return {"test_acc": acc, "lambda": best, "val_acc_by_lambda": {str(k): v for k, v in val_scores.items()},
            "probs": probs}


def save_json(obj, name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {path}", flush=True)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    @property
    def seconds(self):
        return round(time.time() - self.t0, 1)

    def __str__(self):
        return f"{self.seconds}s"
