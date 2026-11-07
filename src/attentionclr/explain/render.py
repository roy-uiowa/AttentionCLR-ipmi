"""Render saliency/CAM heatmaps over input images for the paper figures."""
from __future__ import annotations

import numpy as np
from PIL import Image


def denormalize(x: np.ndarray) -> np.ndarray:
    """Undo the (x - 0.5) / 0.5 normalization used by the dataloader. -> uint8 HWC."""
    x = x * 0.5 + 0.5
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def heatmap_overlay(img_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a [0,1] CAM over an RGB image (HWC uint8) with JET colormap."""
    import cv2

    cam_u8 = (np.clip(cam, 0.0, 1.0) * 255.0).astype(np.uint8)
    cam_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)  # BGR
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * cam_color.astype(np.float32) + (1.0 - alpha) * img_uint8.astype(np.float32))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def saliency_to_uint8(saliency: np.ndarray) -> np.ndarray:
    """Saliency in [0, 1] -> grayscale uint8."""
    return (np.clip(saliency, 0.0, 1.0) * 255.0).astype(np.uint8)
