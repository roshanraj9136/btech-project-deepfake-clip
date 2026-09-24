# Frozen CLIP + Side Network: from DFD-FCG Deepfake Detection to CIFAR-10

B.Tech project by **Roshan Raj**.

This project takes the core idea of **DFD-FCG** (Han et al., *Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model*, CVPR 2025) and tests it on **CIFAR-10**. The idea: keep OpenAI's **CLIP** image encoder completely frozen, and train only a small **side network** that reads CLIP's intermediate layers.

| | |
|---|---|
| **CIFAR-10 test accuracy** | **96.42%** (9,642 / 10,000) |
| Trained weights | 1,788,298 (**2.03%** of the model) |
| Frozen CLIP ViT-B/16 weights | 86,193,152 |
| Training | 5 epochs, 84 min on an RTX 3050 Ti laptop GPU (4 GB) |

The presentation is in [`docs/presentation.pdf`](docs/presentation.pdf).

## Method

```
image 32x32 ──► resize 224x224 (bicubic) ──► CLIP ViT-B/16, 12 layers, FROZEN
                                               │ layer 3  │ layer 6  │ layer 9  │ layer 12
                                               ▼          ▼          ▼          ▼
                          side tokens (0) ──► side 1 ──► side 2 ──► side 3 ──► side 4     TRAINABLE
                                                                                │
                     CLIP final CLS (768) ──────────────┐                        │ side CLS (192)
                                                        ▼                        ▼
                                                  concat 768 + 192 = 960 ──► Linear ──► 10 classes
```

Each **side block** is a small pre-norm transformer block: a linear layer shrinks CLIP's 768-dimensional tokens to 192 and adds them to the side tokens, then 4-head self-attention and a 192 → 384 → 192 feed-forward layer, each with a residual connection (444,672 weights per block).

### What was kept from DFD-FCG and what was changed

| Part | DFD-FCG paper | This project (CIFAR-10) | Why |
|---|---|---|---|
| Input | 10 aligned face frames | 1 image, 32 → 224 px | CIFAR-10 has single images |
| Backbone | CLIP ViT-L/14, frozen | CLIP ViT-B/16, frozen | 4 GB GPU |
| Side network | one block beside each of the 24 layers, full width | 4 blocks at layers 3, 6, 9, 12, width 192 | lighter, same idea |
| Facial component guidance | lips / skin / eyes / nose queries + guidance loss | removed | no faces |
| Temporal branch | frame-to-frame affinity across 10 frames | removed | no video |
| Output / loss | real vs fake, focal loss | 10 classes, cross-entropy | CIFAR-10 labels |

## Results

### CIFAR-10 (10 classes, 10,000 test images)

| Method (same frozen CLIP ViT-B/16) | Trained weights | Test accuracy |
|---|---:|---:|
| Zero-shot CLIP, 1 prompt | 0 | 86.24% |
| Zero-shot CLIP, 10 prompts per class | 0 | 86.74% |
| Linear probe on CLIP's final CLS feature | 7,690 | 95.35% |
| **Frozen CLIP + side network (this project)** | **1,788,298** | **96.42%** |

Per epoch (test accuracy): 94.11 → 95.41 → 95.98 → 96.12 → 96.42%. Final train accuracy 98.37%.
Hardest classes: cat 92.1%, dog 94.7%; cat ↔ dog confusions are 95 of the 358 errors.

A first version that fed the classifier only the side network's averaged tokens (and discarded CLIP's own CLS feature) reached 67.48% after 3 epochs; logs of both runs are in [`results/original_runs/`](results/original_runs/).

Linear probe on the CLS token after each CLIP layer: layer 3 **72.11%**, layer 6 **83.25%**, layer 9 **89.18%**, layer 12 **95.35%**. Object identity lives in CLIP's last layers.

### CIFAKE: real vs AI-generated images (20,000 test images)

To connect the CIFAR-10 experiment back to fake detection, the same model was trained on [CIFAKE](https://arxiv.org/abs/2303.14126): 60,000 real CIFAR-10 photos and 60,000 look-alike images of the same classes generated with latent diffusion.

| Method (same frozen CLIP ViT-B/16) | Accuracy | AUC |
|---|---:|---:|
| Zero-shot CLIP ("a real photo" vs "an AI-generated image") | 49.18% | 0.405 |
| Linear probe, final CLS feature | 95.56% | 0.992 |
| Linear probe, layer 3 / 6 / 9 / 12 | 94.96 / 96.10 / **96.22** / 95.57% | 0.989 / 0.993 / 0.994 / 0.992 |
| **Frozen CLIP + side network, 1 epoch** | **96.92%** | **0.997** |

For real vs fake, the useful signal is strongest in CLIP's **middle** layers (layer 9 beats the last layer), while for object classes the last layer is best by far. A side network that reads several layers beats every single-layer probe. That is the reason DFD-FCG reads intermediate CLIP layers.

Notes: CIFAKE linear probes were trained on a random 30,000-image subset of the training set, and the side network on all 100,000 images for one epoch. Every number comes from a single run.

## Repository layout

```
dfd_fcg_cifar10_experiment.py   model (SideBlock, CLIPWithSideNetworkCIFAR) and the original 5-epoch training run
dfd_fcg_cifar10_768.py          variant with side width 768 (21.3M trainable weights), not trained to completion
experiments/
  common.py                     data loading, CLIP preprocessing, feature extraction, linear probes
  eval_original.py              re-evaluates the trained CIFAR-10 model on the test set
  baselines.py                  zero-shot and linear-probe baselines (cifar10 | cifake)
  train_sidenet.py              faster training script for the same model (cifar10 | cifake)
  run_all.sh                    runs all of the above in order
results/
  cifar10_sidenet_trainable_weights.pth   trained side network + classifier (7 MB)
  cifake_sidenet_seed0_trainable_weights.pth
  *.json                        every reported number, confusion matrices, per-epoch history
  *.npz                         test-set predictions
  logs/, original_runs/         console logs
figures/                        sample images used in the presentation
docs/presentation.pdf           project presentation
```

## How to run

```bash
pip install -r requirements.txt   # install a CUDA build of PyTorch first if you have a GPU

# original training run (downloads CIFAR-10 and CLIP ViT-B/16 on first use)
python dfd_fcg_cifar10_experiment.py

# re-check the trained model: prints 96.42%
cd experiments
python eval_original.py

# baselines and the CIFAKE experiment
python baselines.py cifar10
python baselines.py cifake
python train_sidenet.py cifake --epochs 1 --seed 0
```

The side network weights files hold only the trainable part; CLIP's frozen weights are loaded from timm (`vit_base_patch16_clip_224.openai`).

## Limitations

- ViT-B/16 instead of the paper's ViT-L/14, because of the 4 GB laptop GPU.
- DFD-FCG's two deepfake-specific parts (facial component guidance and the temporal branch) cannot be tested on CIFAR-10.
- CIFAKE contains AI-generated object images, not manipulated face videos.
- One run per setting, so there are no error bars yet.

## References

- Y.-H. Han, T.-M. Huang, K.-L. Hua, J.-C. Chen. *Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model.* CVPR 2025. [arXiv:2404.05583](https://arxiv.org/abs/2404.05583), [code](https://github.com/aiiu-lab/DFD-FCG)
- A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
- J. J. Bird, A. Lotfi. *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.* [arXiv:2303.14126](https://arxiv.org/abs/2303.14126)
- A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images (CIFAR-10).* 2009.
