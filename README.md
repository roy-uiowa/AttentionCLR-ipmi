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
