"""Re-check the trained CIFAR-10 model on the full test set.

Confirms the reported 96.42% and the parameter counts, and saves the confusion matrix
and per-image predictions.

Usage:  python eval_original.py [weights.pth]
The default is results/cifar10_sidenet_trainable_weights.pth (side network + classifier only, 7 MB).
The full 352 MB checkpoint written by dfd_fcg_cifar10_experiment.py also works. CLIP's frozen
weights always come from timm's pretrained 'vit_base_patch16_clip_224.openai'.
"""
import os
import sys

import numpy as np
import torch

from common import (CIFAR10_CLASSES, RESULTS_DIR, CLIPWithSideNetworkCIFAR, Timer, keep_awake,
                    load_arrays, save_json, to_float)

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS_DIR, "cifar10_sidenet_trainable_weights.pth")


def main():
    keep_awake(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer = Timer()
    _, _, x_te, y_te = load_arrays("cifar10")

    model = CLIPWithSideNetworkCIFAR(num_classes=10).to(device)
    missing, unexpected = model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True), strict=False)
    assert not unexpected and all(k.startswith("clip.") for k in missing), (missing, unexpected)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    side = sum(p.numel() for p in model.side_blocks.parameters())
    head = sum(p.numel() for p in model.classifier.parameters())

    probs = []
    with torch.no_grad():
        for i in range(0, len(x_te), 256):
            with torch.amp.autocast("cuda"):
                probs.append(model(to_float(x_te[i:i + 256], device)).float().softmax(1).cpu())
    probs = torch.cat(probs).numpy()
    pred = probs.argmax(1)

    conf = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_te, pred):
        conf[t, p] += 1
    per_class = {c: float(conf[i, i] / conf[i].sum()) for i, c in enumerate(CIFAR10_CLASSES)}
    acc = float((pred == y_te).mean())

    np.savez(os.path.join(RESULTS_DIR, "cifar10_original_preds.npz"), probs=probs, y=y_te)
    save_json({
        "checkpoint": os.path.basename(CKPT),
        "test_images": int(len(y_te)),
        "test_acc": acc,
        "correct": int((pred == y_te).sum()),
        "per_class_acc": per_class,
        "confusion_matrix": conf.tolist(),
        "params": {"trainable_total": trainable, "frozen_clip": frozen,
                   "side_blocks": side, "classifier": head,
                   "trainable_pct": 100 * trainable / (trainable + frozen)},
        "seconds": timer.seconds,
    }, "cifar10_original_checkpoint.json")
    print(f"Test accuracy: {acc * 100:.2f}%  ({timer})", flush=True)
    keep_awake(False)


if __name__ == "__main__":
    main()
