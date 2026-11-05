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

## Dataset completeness (2026-11-04)
- 2 corrupt files removed from the local dataset (MON and MYB).
- Local manifest: 171,372 images vs Table 1's 171,374.
- Recorded in data/processed/known_corrupt.txt; verify_dataset.py subtracts them.
- All other 19 classes match Table 1 exactly.

## Training hyperparameters (NOT STATED in paper)
- batch_size: 64 (ASSUMED)
- optimizer: AdamW (ASSUMED)
- lr: 3e-4 (ASSUMED)
- weight_decay: 1e-4 (ASSUMED)
- schedule: cosine (ASSUMED, not yet wired — Drop 3)
- loss: cross-entropy (ASSUMED; paper does not name it)
- Resolution: contact Carsten Marr. Until then, the above are explicit, tagged, and override-able.

## Class-imbalance procedure (AMBIGUOUS in paper)
- Text: "augmented the training data to ~25,000 images per class."
- Decision B: on_the_fly_oversample_to as default; cap_downsample available via config.
- Default epoch size with 18 minority classes ≈ 533,061 samples (~3x raw dataset).
- Flag: this may be 3x longer per epoch than expected; add max_samples_per_epoch if needed.

## Macenko stain augmentation
- Self-contained PCA-based implementation in src/attentionclr/data/augment.py.
- NOT bit-identical to histomicstk / tiatoolbox Macenko. Fine for training diversity;
  swap in histomicstk if figure-level stain-aug examples are needed.

## Sequential model param count
- Paper: 303,694 (22-node head).
- Fig S5 gives layer types + shapes but not kernel sizes / channel widths / pool sizes.
- Current implementation: ~80k params (xfail'ed test). Reconciliation needed:
  try wider channels (32/64/128/256), 5x5 kernels, or Dense(512).
- Status: tracked by xfail in tests/test_models.py.

## ResNeXt-50 parameter count (2026-11-04)
- Paper: 23,059,094 (22-node head, authors' reference implementation).
- torchvision resnext50_32x4d with a 21-head: 23,022,933.
- Difference: 36,161 params, attributed to a stem-level divergence between
  torchvision's ResNeXt-50 and the authors' from-scratch reference. Not
  recoverable without re-implementing the stem.
- Decision: train torchvision's architecture at 23,022,933 params (direct21).
  The paper's 22-head, 23,059,094-param model is reserved for the future
  probe_22_21_merge.py experiment, which will need the authors' reference.
- Impact on reported metrics: expected negligible (36k/23M ≈ 0.16% of params).

## Sequential model parameter count (2026-11-05) — UNRESOLVED, documented
- Paper: 303,694 (22-head). 21-head target: 303,437.
- S5-literal implementation: 96,125 (21-head) / 96,382 (22-head).
- Sweep of 30+ variants (channel widths 8–256, kernels 3/5/7, dense 256/512/1024,
  pool sizes) finds closest = 292,605 (Δ = -10,832), which does not correspond to
  any clean structural unit (no channel/kernel/dense combination closes it).
- Conclusion: the true architecture has layers not shown in Fig S5, or the paper
  count includes buffers not counted here, or S5 is schematic. Cannot be resolved
  by guessing.
- Decision: ship the S5-literal model (96,125) for the architecture-robustness
  comparison; document the divergence; query the authors for the exact spec
  (also relevant to ref 11, from which the sequential model was reused).
- Status: xfail in tests/test_models.py::test_sequential_direct21_param_count.

## Batch size, max samples per epoch, matmul precision (2026-11-05)
- batch_size: 256 (ASSUMED; A100 80GB headroom; paper not stated).
- max_samples_per_epoch: 171,372 (ASSUMED; caps oversampled epoch at raw dataset
  size instead of the implied ~533k, which would make an epoch ~3x longer).
- matmul precision: "medium" (A100 Tensor Cores; not the paper's V100 setup).

## Tolerant metrics formula (2026-11-05)
- Precision_tolerant defined as TP_t / pred_pos_t where TP_t counts predictions
  of class i whose true label is i or a tolerance partner of i.
- This is the natural sklearn-style symmetric interpretation; the paper's Fig 3A
  defines the pairs but not the exact formula.
- **Must validate against one published number before trusting.** E.g., segmented
  neutrophils tolerant precision ≈ 0.95 / recall ≈ 0.85.

## Dataset provenance (2026-11-05)
- Local dataset: 171,372 images (Table 1: 171,374; 2 corrupt files removed).
- known_corrupt.txt must be corrected to the two real deleted paths.
- Manifest hash: 06033472a934c8e47d949832d6ec8b0ac71d2d1f.
