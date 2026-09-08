# Attention 3D U-Net with Multiple Skip Connections

> Nodirov J., Abdusalomov A.B., Whangbo T.K. **Attention 3D U-Net with Multiple Skip Connections for Segmentation of Brain Tumor Images.** *Sensors* 22(17): 6501, 2022. [doi:10.3390/s22176501](https://doi.org/10.3390/s22176501)

3D U-Net with a **MobileNetV2** encoder, **U-Net 3+** full-scale skip connections, and **attention gates**. Trained on **BraTS-2020** (T1, T1Gd, T2, FLAIR → whole-tumor mask): Adam, lr 0.01, ReduceLROnPlateau, batch size 2, BCE, 200 epochs.

## Architecture

- **Encoder** — 3D MobileNetV2 blocks (§3.2): 1×1×1 expand, 3×3×3 depthwise, 1×1×1 project, ReLU6.
- **Skips** — U-Net 3+ (§3.3): each decoder scale sees every encoder scale plus deeper decoder maps.
- **Attention gates** — 3D AG (§3.4): gating from the coarser map filters the fused skip.
- **Head** — 1×1×1 conv, logits for binary whole-tumor (BCE, §4.3).
- **Norm** — GroupNorm (stable at the paper’s batch size of 2).

Paper input size is 240×240×155.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

BraTS 2020 case folders under `data/brats2020/`, each with `*flair.nii.gz`, `*t1.nii.gz`, `*t1ce.nii.gz`, `*t2.nii.gz`, `*seg.nii.gz`. Whole-tumor label is any non-zero BraTS class.

```bash
PYTHONPATH=src python scripts/train.py --config configs/default.yaml
```

`configs/default.yaml` follows §4.3. Default crop is 128³; set `size: [240, 240, 155]` to match the paper if memory allows.

A small synthetic run (no BraTS download):

```bash
PYTHONPATH=src pytest tests -q
PYTHONPATH=src python scripts/train.py --config configs/small.yaml
```

## Metrics

Dice, precision, recall, Hausdorff — Table 1. Pretrained whole-tumor Dice **0.8974**, HD **5.76**.

## Layout

```
src/attn3dunet/   model, data, train, metrics
experiments/      architecture variants
data/             sample NIfTI volumes
configs/
tests/
scripts/
```

## License

Code: MIT. Paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
