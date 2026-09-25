"""Train the frozen-CLIP + side-network model (same architecture and hyper-parameters as the original run).

Differences from dfd_fcg_cifar10_experiment.py (speed / bookkeeping only, not the method):
  * the whole dataset is held in memory as uint8 and the random flip is done on the GPU
    (the original decoded every image through HuggingFace inside the training loop),
  * a fixed random seed, so the run can be repeated,
  * only the trainable weights are saved (~7 MB instead of 352 MB),
  * per-epoch results are written to results/<run>.json.

Usage:  python train_sidenet.py cifar10 --epochs 5 --seed 0
        python train_sidenet.py cifake  --epochs 3 --seed 0
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from common import (CIFAKE_CLASSES, CIFAR10_CLASSES, RESULTS_DIR, CLIPWithSideNetworkCIFAR, Timer, keep_awake,
                    load_arrays, save_json, to_float)


@torch.no_grad()
def evaluate(model, x, y, device, criterion):
    model.eval()
    probs, loss_sum = [], 0.0
    for i in range(0, len(x), 256):
        yb = torch.from_numpy(y[i:i + 256]).to(device)
        with torch.amp.autocast("cuda"):
            logits = model(to_float(x[i:i + 256], device))
            loss_sum += criterion(logits, yb).item() * len(yb)
        probs.append(logits.float().softmax(1).cpu())
    probs = torch.cat(probs).numpy()
    return loss_sum / len(y), float((probs.argmax(1) == y).mean()), probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["cifar10", "cifake"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    args = ap.parse_args()

    keep_awake(True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = CIFAR10_CLASSES if args.dataset == "cifar10" else CIFAKE_CLASSES
    run = f"{args.dataset}_sidenet_seed{args.seed}"
    timer = Timer()

    x_tr, y_tr, x_te, y_te = load_arrays(args.dataset)
    print(f"[{run}] train {len(y_tr):,}  test {len(y_te):,}", flush=True)

    model = CLIPWithSideNetworkCIFAR(num_classes=len(classes)).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[{run}] trainable {n_trainable:,}  frozen {n_frozen:,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(x_tr))
        seen = correct = 0
        loss_sum = 0.0
        n_batches = int(np.ceil(len(order) / args.batch_size))
        for b in range(n_batches):
            idx = np.sort(order[b * args.batch_size:(b + 1) * args.batch_size])
            x = to_float(x_tr[idx], device)
            flip = torch.rand(len(idx), device=device) < 0.5          # RandomHorizontalFlip(p=0.5)
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
            yb = torch.from_numpy(y_tr[idx]).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            seen += len(idx)
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            if (b + 1) % 100 == 0 or b + 1 == n_batches:
                print(f"  epoch {epoch} batch {b + 1}/{n_batches}  loss {loss_sum / seen:.4f}  "
                      f"train acc {100 * correct / seen:.2f}%  ({timer})", flush=True)
        scheduler.step()

        test_loss, test_acc, probs = evaluate(model, x_te, y_te, device, criterion)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
               "test_loss": test_loss, "test_acc": test_acc, "elapsed_s": timer.seconds}
        if args.dataset == "cifake":
            row["test_auc_fake"] = float(roc_auc_score((y_te == 0).astype(int), probs[:, 0]))
        history.append(row)
        print(f"==> [{run}] epoch {epoch}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items()
                                                          if k != "epoch"), flush=True)
        # keep per-epoch progress on disk, so an interrupted run still leaves its numbers behind
        save_json({"run": run, "args": vars(args), "history": history, "complete": False}, f"{run}.json")

    pred = probs.argmax(1)
    conf = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_te, pred):
        conf[t, p] += 1
    np.savez(os.path.join(RESULTS_DIR, f"{run}_preds.npz"), probs=probs, y=y_te)
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("clip.")},
               os.path.join(RESULTS_DIR, f"{run}_trainable_weights.pth"))
    save_json({
        "run": run, "args": vars(args), "history": history, "complete": True,
        "final_test_acc": history[-1]["test_acc"],
        "per_class_acc": {c: float(conf[i, i] / conf[i].sum()) for i, c in enumerate(classes)},
        "confusion_matrix": conf.tolist(),
        "params": {"trainable": n_trainable, "frozen": n_frozen},
        "seconds": timer.seconds,
    }, f"{run}.json")
    keep_awake(False)


if __name__ == "__main__":
    main()
