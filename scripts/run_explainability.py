"""Generate Figure 5 (SmoothGrad + Grad-CAM) and Figure 6 (UMAP) from a checkpoint.

Usage:
    python scripts/run_explainability.py \
        --config configs/train/paper_5fold.yaml \
        --fold 0 \
        --checkpoint artifacts/runs/fold0/lightning_logs/version_0/checkpoints/epoch=12-step=34801.ckpt \
        --out-dir artifacts/explain/fold0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from attentionclr.data.augment import GeometricConfig, StainConfig
from attentionclr.data.datamodule import BMDataModule, DataConfig
from attentionclr.data.imbalance import ImbalanceConfig
from attentionclr.data.splits import fivefold
from attentionclr.data.table1 import CLASS_NAMES, CLASS_TO_IDX
from attentionclr.explain.gradcam import gradcam, resnext50_target_layer
from attentionclr.explain.render import denormalize, heatmap_overlay, saliency_to_uint8
from attentionclr.explain.smoothgrad import smoothgrad
from attentionclr.explain.umap_features import extract_features, umap_2d
from attentionclr.models.registry import build_model
from attentionclr.train.litmodule import ClassifierModule
from attentionclr.utils.config import load_config, merge_subconfig
from attentionclr.utils.io import read_manifest
from attentionclr.utils.logging import get_logger

log = get_logger("explain")

# Figure 5 example classes. Paper uses 7; we pick the same 7 names where the
# dataset has them. Mapping paper's name -> our CLASS_NAMES entry.
FIG5_CLASSES = [
    "Segmented neutrophils",  # paper: Neutrophil
    "Blasts",                 # paper: Myeloblast (closest match in our schema)
    "Eosinophils",
    "Erythroblasts",
    "Lymphocytes",
    "Plasma cells",
    "Hairy cells",
]


def _build_val_loader(cfg, fold, batch_size):
    from dataclasses import fields
    geo_allowed = {f.name for f in fields(GeometricConfig)}
    a = cfg["augment"]
    dm = BMDataModule(
        DataConfig(
            data_root=Path(cfg["paths"]["data_root"]),
            manifest=Path(cfg["paths"]["manifest"]),
            image_size=cfg.get("image_size", 250),
            num_workers=cfg.get("num_workers", 4),
            batch_size=batch_size,
            geo=GeometricConfig(**{k: v for k, v in a.items() if k in geo_allowed}),
            stain=StainConfig(a.get("stain_method", "macenko"), a.get("stain_sigma", 0.05)),
            imbalance=ImbalanceConfig(
                cfg["training"]["imbalance"]["mode"],
                cfg["training"]["imbalance"]["target_per_class"],
                cfg["training"]["imbalance"].get("max_samples_per_epoch"),
            ),
            seed=cfg.get("seed", 1337),
        ),
        fold=fold,
    )
    dm.setup()
    return dm


def _load_model(cfg, checkpoint, device):
    m = build_model(cfg["model"]["name"], num_classes=cfg["model"].get("num_classes", 21))
    lit = ClassifierModule.load_from_checkpoint(
        checkpoint, model=m,
        lr=cfg["training"].get("lr", 3e-4),
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    return lit.model.eval().to(device)


def _find_one_per_class(model, loader, device, wanted_classes: list[str]) -> dict[str, tuple]:
    """Scan the val loader once; return one correctly-classified example per requested class."""
    wanted_idx = {CLASS_TO_IDX[c]: c for c in wanted_classes}
    found: dict[str, tuple] = {}
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred = logits.argmax(1).cpu()
            for i in range(x.size(0)):
                ci = int(y[i])
                if ci in wanted_idx and int(pred[i]) == ci and wanted_idx[ci] not in found:
                    found[wanted_idx[ci]] = (x[i].cpu(), y[i].item(), pred[i].item())
            if len(found) == len(wanted_idx):
                break
    missing = set(wanted_classes) - set(found)
    if missing:
        log.warning("Could not find correctly classified examples for: %s", sorted(missing))
    return found


def make_figure5(model, loader, device, out_path: Path) -> None:
    """SmoothGrad + Grad-CAM grid: 3 rows x 7 columns."""
    found = _find_one_per_class(model, loader, device, FIG5_CLASSES)
    if not found:
        log.error("Figure 5: no examples found; skipping")
        return

    names = [c for c in FIG5_CLASSES if c in found]
    n = len(names)
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7.5))
    target_layer = resnext50_target_layer(model)

    for j, cls in enumerate(names):
        img_norm, true_y, pred_y = found[cls]
        x = img_norm.unsqueeze(0).to(device)

        # Original (denormalized)
        orig = denormalize(img_norm.permute(1, 2, 0).numpy())
        axes[0, j].imshow(orig)
        axes[0, j].set_title(cls, fontsize=9)
        axes[0, j].axis("off")

        # SmoothGrad
        sal, _ = smoothgrad(model, x, target=torch.tensor([pred_y], device=device))
        sal_u8 = saliency_to_uint8(sal[0].cpu().numpy())
        axes[1, j].imshow(sal_u8, cmap="gray", vmin=0, vmax=255)
        axes[1, j].set_title("SmoothGrad", fontsize=8)
        axes[1, j].axis("off")

        # Grad-CAM
        cam, _ = gradcam(model, target_layer, x, target=torch.tensor([pred_y], device=device))
        overlay = heatmap_overlay(orig, cam[0].cpu().numpy(), alpha=0.45)
        axes[2, j].imshow(overlay)
        axes[2, j].set_title("Grad-CAM", fontsize=8)
        axes[2, j].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def make_figure6(model, loader, device, out_path: Path, max_samples: int = 10_000) -> None:
    """UMAP of penultimate features, colored by true class."""
    # Hook the avgpool output (B, 2048, 1, 1); we flatten to (B, 2048).
    hook_layer = model.model.avgpool
    feats, labels = extract_features(model, loader, hook_layer, device=device, max_samples=max_samples)
    log.info("Extracted %d x %d features", feats.shape[0], feats.shape[1])

    emb = umap_2d(feats, n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42)

    fig, ax = plt.subplots(figsize=(9, 8))
    cmap = plt.get_cmap("tab20", len(CLASS_NAMES))
    for ci in range(len(CLASS_NAMES)):
        mask = labels == ci
        if not mask.any():
            continue
        ax.scatter(emb[mask, 0], emb[mask, 1], s=4, c=[cmap(ci)],
                   label=CLASS_NAMES[ci], alpha=0.7, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=4, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)

    # Also dump the raw embedding for later re-analysis.
    np.savez(out_path.with_suffix(".npz"), embedding=emb, labels=labels)
    log.info("Wrote %s", out_path.with_suffix(".npz"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--samples-fig6", type=int, default=10_000)
    args = ap.parse_args()

    torch.set_float32_matmul_precision("medium")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg["model"] = merge_subconfig(cfg, "model")
    if isinstance(cfg.get("augment"), str):
        cfg["augment"] = merge_subconfig(cfg, "augment")

    manifest = read_manifest(Path(cfg["paths"]["manifest"]))
    folds = fivefold(manifest, seed=cfg.get("seed", 1337), n_splits=cfg["training"]["folds"])
    fold = folds[args.fold]
    dm = _build_val_loader(cfg, fold, batch_size=64)
    val_loader = dm.val_dataloader()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(cfg, args.checkpoint, device)

    log.info("Generating Figure 5 (SmoothGrad + Grad-CAM) ...")
    make_figure5(model, val_loader, device, args.out_dir / "figure5_saliency.png")

    log.info("Generating Figure 6 (UMAP of penultimate features) ...")
    make_figure6(model, val_loader, device, args.out_dir / "figure6_umap.png",
                 max_samples=args.samples_fig6)

    log.info("Done. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()