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
	# pre-commit install || true

data:
data:
	python scripts/build_manifest.py \
		--root data/bone_marrow_cell_dataset \
		--out data/processed/manifest.parquet && \
	python scripts/verify_dataset.py

verify:
	python scripts/verify_dataset.py --known-corrupt data/processed/known_corrupt.txt

smoke:
	python scripts/train.py --config configs/train/paper_5fold.yaml --smoke

test:
	pytest

lint:
	ruff check src tests scripts
	mypy src

fmt:
	ruff format src tests scripts

train-fold0:
	$(PYTHON) scripts/train.py --config configs/train/paper_5fold.yaml --fold 0

train-all:
	for f in 0 1 2 3 4; do \
		$(PYTHON) scripts/train.py --config configs/train/paper_5fold.yaml --fold $$f || exit 1; \
	done

evaluate-all:
	for f in 0 1 2 3 4; do \
		$(PYTHON) scripts/evaluate.py --config configs/train/paper_5fold.yaml \
			--fold $$f \
			--checkpoint artifacts/runs/fold$$f/checkpoints/last.ckpt \
			--run-dir artifacts/runs || exit 1; \
	done

table1:
	$(PYTHON) scripts/make_table1.py

clean:
	rm -rf artifacts/runs/* .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
