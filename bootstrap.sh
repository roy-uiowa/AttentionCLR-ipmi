#!/usr/bin/env bash
# bootstrap.sh — create the attentionCLR project skeleton.
# Safe to re-run: uses mkdir -p and only creates files that don't already exist.
set -euo pipefail

ROOT="/Users/troy3/myProjects/attentionCLR"
cd "$ROOT"

echo "==> Creating directory tree"
mkdir -p \
  data/processed \
  data/external/choi2017 \
  configs/models configs/train configs/augment configs/eval \
  src/attentionclr/data \
  src/attentionclr/models \
  src/attentionclr/losses \
  src/attentionclr/train \
  src/attentionclr/eval \
  src/attentionclr/explain \
  src/attentionclr/utils \
  scripts \
  notebooks \
  tests \
  artifacts/runs artifacts/figures artifacts/tables \
  reports

echo "==> Creating __init__.py files"
for pkg in \
  src/attentionclr \
  src/attentionclr/data \
  src/attentionclr/models \
  src/attentionclr/losses \
  src/attentionclr/train \
  src/attentionclr/eval \
  src/attentionclr/explain \
  src/attentionclr/utils ; do
  touch "$pkg/__init__.py"
done

echo "==> Writing .gitignore"
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/

# Data (keep the raw dataset and processed manifest out of git)
data/bone_marrow_cell_dataset/
data/external/
data/processed/*
!data/processed/label_map.yaml
!data/processed/tolerance_classes.yaml
!data/processed/.gitkeep

# Artifacts
artifacts/
reports/_build/

# OS
.DS_Store
Thumbs.db

# Editor
.vscode/
.idea/
EOF

echo "==> Writing placeholder data/processed/.gitkeep"
touch data/processed/.gitkeep

echo "==> Writing label_map.yaml (CONFIRM the codes before trusting this)"
cat > data/processed/label_map.yaml <<'EOF'
# Canonical mapping: dataset folder code -> paper Table 1 class name.
# Provenance: inferred from Matek et al. 2021, Blood 138(20):1917-1927, Table 1
# and German hematology atlas conventions. VERIFY before training.
ABE: Abnormal eosinophils
ART: Artefacts
BAS: Basophils
BLA: Blasts
EBO: Erythroblasts
EOS: Eosinophils
FGC: Faggot cells
HAC: Hairy cells
KSC: Smudge cells
LYI: Immature lymphocytes
LYT: Lymphocytes
MMZ: Metamyelocytes
MON: Monocytes
MYB: Myelocytes
NGB: Band neutrophils
NGS: Segmented neutrophils
NIF: Not identifiable
OTH: Other cells
PEB: Proerythroblasts
PLM: Plasma cells
PMO: Promyelocytes
EOF

echo "==> Writing tolerance_classes.yaml (PLACEHOLDER — replace after digitizing Fig 3A)"
cat > data/processed/tolerance_classes.yaml <<'EOF'
# Tolerant-evaluation pairs for strict/tolerant precision/recall.
# PLACEHOLDER. The paper encodes these in Figure 3A (blue off-diagonal cells)
# but does not tabulate them. Replace with the digitized list before reporting
# tolerant metrics. Each entry: [class_a, class_b] means a prediction of either
# counts as correct for the other under tolerant evaluation.
tolerance_pairs: []
# Example (uncomment + verify against Fig 3A):
# tolerance_pairs:
#   - [Band neutrophils, Segmented neutrophils]
#   - [Myelocytes, Metamyelocytes]
#   - [Proerythroblasts, Erythroblasts]
EOF

echo "==> Writing pyproject.toml"
cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "attentionclr"
version = "0.1.0"
description = "Reproduction + extension of Matek et al. 2021 (Blood) bone marrow cell classification."
requires-python = ">=3.10"
dependencies = [
  "torch>=2.2",
  "torchvision>=0.17",
  "lightning>=2.2",
  "timm>=0.9",
  "torchmetrics>=1.3",
  "numpy>=1.26",
  "pandas>=2.1",
  "pyarrow>=14.0",
  "scikit-learn>=1.4",
  "pyyaml>=6.0",
  "pillow>=10.0",
  "opencv-python-headless>=4.8",
  "matplotlib>=3.8",
  "seaborn>=0.13",
  "umap-learn>=0.5",
  "tqdm>=4.66",
  "rich>=13.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=4.1",
  "ruff>=0.3",
  "mypy>=1.9",
  "pre-commit>=3.6",
  "ipykernel>=6.29",
  "jupyterlab>=4.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
EOF

echo "==> Writing Makefile"
cat > Makefile <<'EOF'
.PHONY: help install data verify smoke test lint fmt clean

help:
	@echo "Targets:"
	@echo "  install   pip install -e '.[dev]'"
	@echo "  data      build manifest + verify dataset counts"
	@echo "  verify    assert dataset matches Table 1 counts"
	@echo "  smoke     2-min CPU smoke train on 100 images"
	@echo "  test      run pytest"
	@echo "  lint      ruff + mypy"
	@echo "  fmt       ruff format"
	@echo "  clean     remove caches + artifacts/runs"

install:
	pip install -e ".[dev]"
	pre-commit install || true

data:
	python scripts/build_manifest.py --root data/bone_marrow_cell_dataset --out data/processed
	python scripts/verify_dataset.py

verify:
	python scripts/verify_dataset.py

smoke:
	python scripts/train.py --config configs/train/paper_5fold.yaml --smoke

test:
	pytest

lint:
	ruff check src tests scripts
	mypy src

fmt:
	ruff format src tests scripts

clean:
	rm -rf artifacts/runs/* .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
EOF

echo "==> Writing .pre-commit-config.yaml"
cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.5
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml
      - id: check-added-large-files
        args: ["--maxkb=2000"]
EOF

echo "==> Writing config stubs"
cat > configs/default.yaml <<'EOF'
# Shared defaults. Every run loads this, then overlays the model/train/augment/eval config.
paths:
  data_root: data/bone_marrow_cell_dataset
  manifest: data/processed/manifest.parquet
  label_map: data/processed/label_map.yaml
  tolerance_classes: data/processed/tolerance_classes.yaml
  artifacts: artifacts

seed: 1337
num_workers: 4
image_size: 250
num_classes: 21
num_output_nodes: 22        # paper: 22 nodes, 2 combined -> 21 (see assumptions.md)
merge_21_from_22: null      # TODO: resolve which two nodes merge
EOF

cat > configs/models/resnext50.yaml <<'EOF'
# ResNeXt-50, C=32, 250x250 input, 22->21 output. Target: 23,059,094 params.
name: resnext50
arch: resnext50_32x4d
pretrained: false
in_channels: 3
num_output_nodes: 22
merge_to: 21
expected_params: 23059094
EOF

cat > configs/models/sequential.yaml <<'EOF'
# Sequential CNN per Supplement Fig S5. Target: 303,694 params.
name: sequential
arch: sequential_s5
in_channels: 3
num_output_nodes: 22
merge_to: 21
expected_params: 303694
# Assumptions (documented in reports/assumptions.md):
conv_kernel: 3
conv_activation: relu
pool_sizes: [4, 4, 5]      # inferred from output shapes 250->62->15->3
pool_strides: [4, 4, 5]
EOF

cat > configs/train/paper_5fold.yaml <<'EOF'
# Matek et al. 2021, 5-fold stratified CV, 80/20 per fold, 13 epochs.
defaults: [../default.yaml]
model: ../models/resnext50.yaml
augment: ../augment/paper_aug.yaml
eval: ../eval/tolerant.yaml
training:
  folds: 5
  epochs: 13
  train_frac: 0.8
  batch_size: null          # NOT STATED IN PAPER — set explicitly; log as assumption
  optimizer: null           # NOT STATED — set explicitly; log as assumption
  lr: null
  schedule: null
  weight_decay: null
  loss: null
  imbalance:
    mode: on_the_fly_oversample_to  # see reports/assumptions.md §imbalance
    target_per_class: 25000
  precision: 32
  devices: auto
EOF

cat > configs/train/paper_80_20.yaml <<'EOF'
# Single 80/20 split (main text "Methods" primary protocol).
defaults: [../default.yaml]
model: ../models/resnext50.yaml
augment: ../augment/paper_aug.yaml
eval: ../eval/tolerant.yaml
training:
  folds: 1
  epochs: 13
  train_frac: 0.8
  batch_size: null
  optimizer: null
  lr: null
  schedule: null
  weight_decay: null
  loss: null
  imbalance:
    mode: on_the_fly_oversample_to
    target_per_class: 25000
  precision: 32
  devices: auto
EOF

cat > configs/train/robustness_70_30.yaml <<'EOF'
# Robustness check vs Krappe's 70/30 split (Supp Fig S6).
defaults: [../default.yaml]
model: ../models/resnext50.yaml
augment: ../augment/paper_aug.yaml
eval: ../eval/tolerant.yaml
training:
  folds: 1
  epochs: 13
  train_frac: 0.7
  batch_size: null
  optimizer: null
  lr: null
  schedule: null
  weight_decay: null
  loss: null
  imbalance:
    mode: on_the_fly_oversample_to
    target_per_class: 25000
  precision: 32
  devices: auto
EOF

cat > configs/augment/paper_aug.yaml <<'EOF'
# Geometric + stain augmentation per Matek 2021 Methods.
geometric:
  rotation_deg: [0, 180]
  hflip: true
  vflip: true
  shift_frac: 0.25
  shear_frac: 0.20
stain:
  method: macenko
  # Tellez et al. default perturbation strength (paper cites ref 33)
  sigma: 0.05
EOF

cat > configs/eval/tolerant.yaml <<'EOF'
# Strict + tolerant precision/recall (paper Table 1 / Fig 3B).
metrics: [strict_precision, strict_recall, tolerant_precision, tolerant_recall]
tolerance_classes: data/processed/tolerance_classes.yaml
report_per_fold: true
report_mean_std: true
EOF

echo "==> Writing stub modules (so imports work and tests can be filled in)"
cat > src/attentionclr/data/manifest.py <<'EOF'
"""Build and read the dataset manifest. See scripts/build_manifest.py."""
EOF
cat > src/attentionclr/data/dataset.py <<'EOF'
"""torch Dataset + LightningDataModule over the manifest."""
EOF
cat > src/attentionclr/data/splits.py <<'EOF'
"""Stratified k-fold, 80/20, 70/30 splitters (manifest in -> split indices out)."""
EOF
cat > src/attentionclr/data/augment.py <<'EOF'
"""Geometric + Macenko stain augmentation per Matek 2021."""
EOF
cat > src/attentionclr/data/imbalance.py <<'EOF'
"""On-the-fly oversampling to ~25k/class (assumption; see reports/assumptions.md)."""
EOF
cat > src/attentionclr/models/resnext50.py <<'EOF'
"""ResNeXt-50 (C=32), 250x250 in, 22->21 out. Target: 23,059,094 params."""
EOF
cat > src/attentionclr/models/sequential.py <<'EOF'
"""Sequential CNN per Supplement Fig S5. Target: 303,694 params."""
EOF
cat > src/attentionclr/models/registry.py <<'EOF'
"""Name -> model factory."""
EOF
cat > src/attentionclr/losses/cross_entropy.py <<'EOF'
"""Cross-entropy (paper loss unspecified; assumption)."""
EOF
cat > src/attentionclr/losses/class_balanced.py <<'EOF'
"""Optional class-balanced loss (extension)."""
EOF
cat > src/attentionclr/losses/supcon.py <<'EOF'
"""Supervised contrastive loss (CLR extension)."""
EOF
cat > src/attentionclr/train/trainer.py <<'EOF'
"""LightningModule + Trainer wrapper; seeding, provenance, checkpointing."""
EOF
cat > src/attentionclr/train/callbacks.py <<'EOF'
"""Provenance, config-hash, metric-logging callbacks."""
EOF
cat > src/attentionclr/train/optim.py <<'EOF'
"""Optimizer/scheduler factories."""
EOF
cat > src/attentionclr/eval/metrics.py <<'EOF'
"""Strict + tolerant precision/recall, matching Table 1."""
EOF
cat > src/attentionclr/eval/confusion.py <<'EOF'
"""Row-normalized confusion matrices (paper Figure 2 style)."""
EOF
cat > src/attentionclr/eval/external_validation.py <<'EOF'
"""Choi 2017 external-validation pipeline + abstention rate."""
EOF
cat > src/attentionclr/explain/smoothgrad.py <<'EOF'
"""SmoothGrad saliency (Smilkov et al.)."""
EOF
cat > src/attentionclr/explain/gradcam.py <<'EOF'
"""Grad-CAM saliency (Selvaraju et al.)."""
EOF
cat > src/attentionclr/explain/umap_features.py <<'EOF'
"""2048-d penultimate features -> UMAP (paper Figure 6)."""
EOF
cat > src/attentionclr/utils/seed.py <<'EOF'
"""Deterministic seeding (python, numpy, torch, cuda)."""
EOF
cat > src/attentionclr/utils/io.py <<'EOF'
"""YAML/Parquet I/O helpers."""
EOF
cat > src/attentionclr/utils/logging.py <<'EOF'
"""Rich logging setup."""
EOF
cat > src/attentionclr/utils/provenance.py <<'EOF'
"""Capture git SHA, dirty flag, config hash, env into each run dir."""
EOF

echo "==> Writing script stubs"
cat > scripts/build_manifest.py <<'EOF'
"""Scan data/bone_marrow_cell_dataset -> data/processed/manifest.parquet."""
EOF
cat > scripts/verify_dataset.py <<'EOF'
"""Assert per-class counts match Matek 2021 Table 1. Fails loudly on mismatch."""
EOF
cat > scripts/audit_labels.py <<'EOF'
"""Check label spellings (Fagott/Faggot, Promyelocytes, etc.) and folder/file consistency."""
EOF
cat > scripts/train.py <<'EOF'
"""Entry point: --config <yaml> [--smoke] [--fold N]."""
EOF
cat > scripts/evaluate.py <<'EOF'
"""Evaluate a checkpoint on a split; emit strict+tolerant metrics JSON."""
EOF
cat > scripts/make_table1.py <<'EOF'
"""Regenerate Table 1 (mean +/- SD across folds) from artifacts/runs."""
EOF
cat > scripts/make_fig2_fig3.py <<'EOF'
"""Confusion matrix (Fig 2) + model comparison (Fig 3B)."""
EOF
cat > scripts/run_explainability.py <<'EOF'
"""SmoothGrad + Grad-CAM + UMAP figures."""
EOF
cat > scripts/probe_22_21_merge.py <<'EOF'
"""Experiment: enumerate candidate 22->21 merges; compare against reported behavior."""
EOF

echo "==> Writing test stubs"
cat > tests/test_manifest.py <<'EOF'
"""Dataset integrity: per-class counts match Table 1."""
EOF
cat > tests/test_models.py <<'EOF'
"""Param-count tests: resnext50 == 23,059,094; sequential == 303,694."""
EOF
cat > tests/test_metrics.py <<'EOF'
"""Strict vs tolerant precision/recall on toy data."""
EOF
cat > tests/test_splits.py <<'EOF'
"""No leakage; stratification preserved."""
EOF
cat > tests/test_smoke_train.py <<'EOF'
"""2-min CPU smoke train on 100 images; produces a valid run dir."""
EOF

echo "==> Writing reports stubs"
cat > reports/reproduction_status.md <<'EOF'
# Reproduction Status

Living document. One row per paper artifact (Table 1 row, Figure, claim).

| Artifact | Status | Notes |
|---|---|---|
| Table 1 counts | TODO | verify via scripts/verify_dataset.py |
| Table 1 strict metrics | TODO | |
| Table 1 tolerant metrics | TODO | depends on tolerance_classes.yaml |
| Figure 2 confusion matrix | TODO | |
| Figure 3B model comparison | TODO | |
| Figure 4 external (Choi) | TODO | Choi data availability? |
| Fig S4 sequential external | TODO | target abstention 207/627 |
| Figure 5 saliency | TODO | qualitative |
| Figure 6 UMAP | TODO | qualitative |
EOF

cat > reports/assumptions.md <<'EOF'
# Assumptions

Every value not explicitly stated in Matek et al. 2021 (main text or supplement)
lives here, with the reasoning and the experiment that would resolve it.

## Training hyperparameters (NOT STATED)
- batch_size, optimizer, lr, schedule, weight_decay, loss: **unknown**
- Resolution: email Carsten Marr (carsten.marr@helmholtz-muenchen.de); until then,
  set in configs/train/*.yaml and tag with `# ASSUMED`.

## 22 -> 21 output merge (NOT STATED)
- Paper: 22 output nodes, 2 combined -> 21 classes.
- Supplement Fig S5 shows Dense(22) -> activation(21) in the sequential model.
- Which two nodes merge: **unknown**.
- Experiment: scripts/probe_22_21_merge.py enumerates candidate pairs, evaluates
  each trained 22-output model, and compares to reported ResNeXt behavior.

## Class-imbalance procedure (AMBIGUOUS)
- Text: "augmented the training data to ~25,000 images per class."
- Three readings: pre-generated oversample; on-the-fly oversample; cap+downsample.
- Default: on-the-fly oversample to 25k. Config flag allows alternatives.

## Sequential architecture (PARTIALLY STATED)
- Fig S5 gives layer types + output shapes, but not kernel sizes, activations, or
  pool/stride/padding. Inferred: 3x3 kernels, ReLU, pool sizes [4,4,5].
- Hard check: total params must equal 303,694 (tests/test_models.py).

## Tolerance-class pairs (ENCODED IN FIGURE ONLY)
- Fig 3A blue off-diagonal; not tabulated. Digitize before reporting tolerant metrics.

## Augmentation details
- Rotation is "0 to 180" (not +/-180). Shift 25%, shear 20%. Stain aug via
  Macenko + Tellez defaults (sigma assumed 0.05).
EOF

cat > reports/data_dictionary.md <<'EOF'
# Data Dictionary

| Folder code | Table 1 class name | Count (Table 1) |
|---|---|---:|
| ABE | Abnormal eosinophils | 8 |
| ART | Artefacts | 19630 |
| BAS | Basophils | 441 |
| BLA | Blasts | 11973 |
| EBO | Erythroblasts | 27395 |
| EOS | Eosinophils | 5883 |
| FGC | Faggot cells | 47 |
| HAC | Hairy cells | 409 |
| KSC | Smudge cells | 42 |
| LYI | Immature lymphocytes | 65 |
| LYT | Lymphocytes | 26242 |
| MMZ | Metamyelocytes | 3055 |
| MON | Monocytes | 4040 |
| MYB | Myelocytes | 6557 |
| NGB | Band neutrophils | 9968 |
| NGS | Segmented neutrophils | 29424 |
| NIF | Not identifiable | 3538 |
| OTH | Other cells | 294 |
| PEB | Proerythroblasts | 2740 |
| PLM | Plasma cells | 7629 |
| PMO | Promyelocytes | 11994 |
| **Total** | | **171374** |
EOF

cat > README.md <<'EOF'
# attentionCLR

Reproduction of Matek et al., *Blood* 2021, "Highly accurate differentiation of
bone marrow cell morphologies using deep neural networks on a large image data
set", with a scaffold for attention / contrastive-learning extensions.

## Quickstart

```bash
make install
make data      # build manifest + verify counts against Table 1
make verify    # re-run the count check
make smoke     # 2-min CPU smoke train
make test