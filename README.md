# Attention 3D U-Net with Multiple Skip Connections

Implementation of:

> Nodirov J., Abdusalomov A.B., Whangbo T.K. **Attention 3D U-Net with Multiple Skip Connections for Segmentation of Brain Tumor Images.** *Sensors* 22(17): 6501, 2022. [doi:10.3390/s22176501](https://doi.org/10.3390/s22176501)

3D U-Net encoder built from **MobileNetV2 inverted residuals**, **U-Net 3+ full-scale skip connections**, and **attention gates** on the skips. Trained in the paper on **BraTS-2020** (T1, T1Gd, T2, FLAIR → whole-tumor mask) with Adam, lr 0.01, ReduceLROnPlateau, batch size 2, BCE, 200 epochs.

This repo is a from-scratch PyTorch rewrite of that architecture. It is not the original training dump.

## Architecture

- **Encoder** — 3D MobileNetV2 blocks (§3.2): 1×1×1 expand, 3×3×3 depthwise, 1×1×1 project, ReLU6, residual when shapes match.
- **Skips** — U-Net 3+ (§3.3): each decoder scale sees every encoder scale (pooled or upsampled) plus deeper decoder maps.
- **Attention gates** — Oktay-style 3D AG (§3.4): gating from the coarser map filters the fused skip.
- **Head** — 1×1×1 conv, logits for binary whole-tumor (BCE as in §4.3).
- **Norm** — GroupNorm instead of BatchNorm so batch size 1–2 (the paper’s batch size) is stable.

Paper input size is 240×240×155. That needs a GPU. CPU tests use 16³ and a thin width multiplier.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# CPU-only torch if you do not have CUDA:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## CPU (this machine)

```bash
PYTHONPATH=src pytest tests/test_cpu.py -q
PYTHONPATH=src python scripts/train.py --config configs/cpu.yaml
```

Synthetic 4-channel volumes, 16³, three epochs. Confirms the graph, loss, checkpoint. Not a Dice number from the paper.

## GPU + BraTS-2020

1. Download [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/) and unpack case folders under `data/brats2020/`.
2. Each case needs `*flair.nii.gz`, `*t1.nii.gz`, `*t1ce.nii.gz`, `*t2.nii.gz`, `*seg.nii.gz`.
3. Whole-tumor label = any non-zero BraTS class (NCR/NET, ED, ET), matching the paper’s binary BCE setup.

```bash
PYTHONPATH=src python scripts/train.py --config configs/default.yaml
```

`configs/default.yaml` follows §4.3: 200 epochs, batch 2, Adam, lr 0.01, scheduler factor 0.2 / patience 5. Volumes are resampled to 128³ so a single 24 GB GPU can hold batch 2; change `size` to `[240, 240, 155]` if you have the memory the paper used.

## Metrics

Dice, precision, recall, Hausdorff — same four numbers as Table 1. Paper (pretrained) whole-tumor Dice **0.8974**, HD **5.76**. Reproducing that needs BraTS + GPU; do not expect it from the CPU synthetic run.

## Layout

```
src/attn3dunet/   model, MobileNetV2-3D, attention, data, train, metrics
configs/          cpu.yaml · default.yaml
tests/            CPU unit tests
scripts/          train.py · infer.py
```

## License

Code: MIT. Paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## History

This GitHub repository was created on **20 February 2022** as `3dunt_simple`,
during work on the paper (submitted August 2022). The original experiment
scripts, sample NIfTI volumes, and checkpoints live in [`legacy/`](legacy/).
The current `src/attn3dunet` tree is a cleaned reimplementation of the
published architecture on that same history.
