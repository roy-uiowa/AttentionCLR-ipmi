"""Geometric + Macenko stain augmentation, per Matek 2021 Methods.

Geometric (from paper):
    - random rotation in [0, 180] degrees
    - vertical and horizontal flips
    - shifts up to 25% of image width/height
    - shears up to 20%

Stain (from paper, refs 32-34):
    - Macenko PCA decomposition into eosin/hematoxylin
    - perturb each component with Gaussian noise (Tellez et al. defaults)
    - recompose

The Macenko implementation here is self-contained (no extra deps beyond
numpy + cv2) and deterministic given a seed. It is applied AFTER the
train/val split (as in the paper).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometricConfig:
    rotation_deg: tuple[float, float] = (0.0, 180.0)
    hflip: bool = True
    vflip: bool = True
    shift_frac: float = 0.25
    shear_frac: float = 0.20


@dataclass(frozen=True)
class StainConfig:
    method: str = "macenko"
    sigma: float = 0.05          # Tellez et al. default perturbation strength
    alpha: float = 1.0           # Macenko percentile threshold (standard: 1.0)
    beta: float = 0.15           # Macenko OD threshold (standard: 0.15)
    luminosity_threshold: float = 0.8


def geometric_augment(
    img: np.ndarray,  # HxWx3 uint8
    cfg: GeometricConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    h, w = img.shape[:2]

    # Affine: rotation + shear + shift in one matrix, then warp.
    angle = float(rng.uniform(*cfg.rotation_deg))
    shear_x = float(rng.uniform(-cfg.shear_frac, cfg.shear_frac))
    shear_y = float(rng.uniform(-cfg.shear_frac, cfg.shear_frac))
    tx = float(rng.uniform(-cfg.shift_frac, cfg.shift_frac)) * w
    ty = float(rng.uniform(-cfg.shift_frac, cfg.shift_frac)) * h

    cx, cy = w / 2.0, h / 2.0
    # Build rotation+shear around center, then translate.
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    shear = np.array(
        [[1.0, shear_x, -shear_x * cy], [shear_y, 1.0, -shear_y * cx]], dtype=np.float64
    )
    # Compose: shear @ [rot; 0 0 1]
    rot_h = np.vstack([rot, [0.0, 0.0, 1.0]])
    shear_h = np.vstack([shear, [0.0, 0.0, 1.0]])
    affine = (shear_h @ rot_h)[:2]
    affine[0, 2] += tx
    affine[1, 2] += ty

    out = cv2.warpAffine(img, affine, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if cfg.hflip and rng.random() < 0.5:
        out = out[:, ::-1]
    if cfg.vflip and rng.random() < 0.5:
        out = out[::-1, :]
    return np.ascontiguousarray(out)


# --- Macenko stain augmentation ------------------------------------------------

def _rgb_to_od(img: np.ndarray) -> np.ndarray:
    """Optical density, standard transform: OD = -log10((I + 1) / 256)."""
    return -np.log10((img.astype(np.float64) + 1.0) / 256.0)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    rgb = 256.0 * np.power(10.0, -od) - 1.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _macenko_decompose(
    img: np.ndarray, cfg: StainConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (stain_matrix [3x2], concentrations [HxWx2], background_od [3])."""
    od = _rgb_to_od(img).reshape(-1, 3)
    # Remove transparent pixels (low OD)
    mask = (np.sum(od, axis=1) > cfg.beta) & (od.max(axis=1) > 0.15)
    od_hat = od[mask]
    if od_hat.shape[0] < 10:
        raise ValueError("Too few non-background pixels for Macenko decomposition")

    # Eigen-decomposition of OD covariance
    _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    # Project onto the two principal directions
    proj = od_hat @ eigvecs[:, -2:]
    # Find extreme angles in projected space
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    a_min, a_max = np.percentile(angles, (cfg.alpha, 100 - cfg.alpha))
    v_min = eigvecs[:, -2:] @ np.array([np.cos(a_min), np.sin(a_min)])
    v_max = eigvecs[:, -2:] @ np.array([np.cos(a_max), np.sin(a_max)])
    stain_matrix = np.stack([v_min, v_max], axis=1)  # 3x2
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=0, keepdims=True)

    # Solve for concentrations: OD = stain_matrix @ C  =>  C = pinv(SM) @ OD
    concentrations = od @ np.linalg.pinv(stain_matrix).T  # (H*W) x 2
    background_od = np.percentile(od, 99, axis=0)
    return stain_matrix, concentrations.reshape(*img.shape[:2], 2), background_od


def macenko_augment(
    img: np.ndarray,
    cfg: StainConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb eosin/hematoxylin concentrations by Gaussian noise; recompose."""
    try:
        stain_matrix, concentrations, background_od = _macenko_decompose(img, cfg)
    except Exception:
        # If decomposition fails (rare; low-stain patches), return original unchanged.
        # This is safer than crashing training on a single bad patch.
        return img

    # Perturb each stain concentration independently (Tellez et al. approach)
    noise = rng.normal(0.0, cfg.sigma, size=concentrations.shape)
    concentrations = np.clip(concentrations * (1.0 + noise), 0.0, None)

    od_reconstructed = concentrations @ stain_matrix.T
    return _od_to_rgb(od_reconstructed.reshape(img.shape))


def augment_image(
    img: np.ndarray,
    geo_cfg: GeometricConfig,
    stain_cfg: StainConfig,
    rng: np.random.Generator,
    apply_stain: bool = True,
) -> np.ndarray:
    out = geometric_augment(img, geo_cfg, rng)
    if apply_stain:
        out = macenko_augment(out, stain_cfg, rng)
    return out
