# Experiments

The paper model is `configs/experiments/paper.yaml`. Everything else is a trial that may land **better or worse**. No scores here until BraTS + GPU runs exist.

Run one:

```bash
PYTHONPATH=src python scripts/train.py --config configs/experiments/bce_dice.yaml
```

| Config | What changes | Why try it |
|---|---|---|
| `paper` | MobileNetV2 encoder, full-scale skips, attention gates, BCE | Published method (§3–4) |
| `no_attention` | Same, skip filter off | Paper Table 3 ablation |
| `no_fullscale` | Same-scale U-Net skips only | Paper Table 3 “no skip connections” (stricter: no *full-scale* skips) |
| `unet3d` | Plain 3D U-Net, no MobileNet | Table 1 baseline |
| `bce_dice` | Paper net, 0.5 BCE + 0.5 Dice | Tumors are sparse; BCE alone often under-segments |
| `focal` | Paper net, focal loss | Down-weight easy background voxels |
| `tversky` | Paper net, Tversky β=0.7 | Penalize missed tumor more than extra false positives (paper Fig. 11) |
| `se_skip` | Squeeze-and-excitation instead of attention gates | Cheaper channel filter; may or may not match AG |
| `trans_bottleneck` | MHSA on the bottleneck | Paper §6 future work (transformers) |
| `deep_supervision` | Extra decoder heads + aux loss | U-Net 3+ training trick, not in the paper |
| `augment` | Random flips + intensity jitter | Paper §6 said classic aug did not help; re-check |
| `dropout` | 3D dropout 0.2 in the bottleneck | Cheap uncertainty / regulariser |

All GPU configs inherit `configs/default.yaml` (200 epochs, batch 2, BraTS path). Override `device` / `synthetic` locally if needed.
