"""BraTS-2020 loader and a synthetic dataset for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("flair", "t1", "t1ce", "t2")


def _ellipsoid(shape: Sequence[int], center, radii) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    return (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    ) <= 1.0


class SyntheticBraTS(Dataset):
    """Four-channel noisy MRI with an ellipsoid whole-tumor mask.

    Enough to exercise the pipeline without BraTS. Not a substitute for BraTS.
    """

    def __init__(self, n: int = 8, size: Tuple[int, int, int] = (16, 16, 16), seed: int = 0) -> None:
        self.n = n
        self.size = size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(self.rng.integers(0, 1_000_000) + idx)
        d, h, w = self.size
        image = rng.normal(0.0, 0.15, size=(4, d, h, w)).astype(np.float32)
        center = (d * 0.5, h * 0.5, w * 0.5)
        radii = (max(d * 0.25, 2), max(h * 0.3, 2), max(w * 0.28, 2))
        mask = _ellipsoid((d, h, w), center, radii).astype(np.float32)
        image[:, mask.astype(bool)] += 0.8
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask[None]),
        }


class RandomFlipIntensity:
    """Axis flips + small intensity jitter. Paper §6 said classic aug did not help;
    this trial checks that again with a light 3D set."""

    def __init__(self, inner: Dataset) -> None:
        self.inner = inner

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict:
        item = dict(self.inner[idx])
        img, mask = item["image"], item["mask"]
        for dim in (1, 2, 3):
            if torch.rand(()) > 0.5:
                img = torch.flip(img, [dim])
                mask = torch.flip(mask, [dim])
        scale = 0.9 + 0.2 * float(torch.rand(()))
        img = img * scale + 0.05 * torch.randn_like(img)
        item["image"] = img
        item["mask"] = mask
        return item


class BraTS2020(Dataset):
    """Folder-per-case BraTS 2020 layout.

    Each case directory should contain ``*flair.nii.gz``, ``*t1.nii.gz``,
    ``*t1ce.nii.gz``, ``*t2.nii.gz``, and ``*seg.nii.gz``. Whole-tumor mask
    is any non-zero label (NCR/NET, ED, ET), matching the paper's BCE setup.
    """

    def __init__(self, root, size: Optional[Tuple[int, int, int]] = (128, 128, 128)) -> None:
        self.root = Path(root)
        self.size = size
        self.cases = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not self.cases:
            raise FileNotFoundError(f"No BraTS case folders under {self.root}")

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> dict:
        import nibabel as nib

        case = self.cases[idx]
        channels = []
        for mod in MODALITIES:
            matches = list(case.glob(f"*{mod}.nii*"))
            if not matches:
                raise FileNotFoundError(f"{mod} missing in {case}")
            vol = nib.load(str(matches[0])).get_fdata().astype(np.float32)
            vol = (vol - vol.mean()) / (vol.std() + 1e-6)
            channels.append(vol)
        image = np.stack(channels, axis=0)
        seg_path = list(case.glob("*seg.nii*"))
        if not seg_path:
            raise FileNotFoundError(f"seg missing in {case}")
        seg = nib.load(str(seg_path[0])).get_fdata()
        mask = (seg > 0).astype(np.float32)[None]
        image_t = torch.from_numpy(image)
        mask_t = torch.from_numpy(mask)
        if self.size is not None:
            image_t = torch.nn.functional.interpolate(
                image_t[None], size=self.size, mode="trilinear", align_corners=False
            )[0]
            mask_t = torch.nn.functional.interpolate(
                mask_t[None], size=self.size, mode="nearest"
            )[0]
        return {"image": image_t, "mask": mask_t}
