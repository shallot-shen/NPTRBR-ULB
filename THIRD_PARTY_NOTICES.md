# Data and method attribution

This repository imports the official TabM and numerical-embedding packages. It
does not redistribute their source files, wheels or weights, and is not an
official release from their authors. The local legacy model and the official
backbone are separate recipes.

- [TabM](https://github.com/yandex-research/tabm): Y. Gorishniy, A. Kotelnikov and A. Babenko, *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*, ICLR 2025, [arXiv:2410.24210](https://arxiv.org/abs/2410.24210).
- [Numerical embeddings](https://github.com/yandex-research/rtdl-num-embeddings): Y. Gorishniy, I. Rubachev and A. Babenko, *On Embeddings for Numerical Features in Tabular Deep Learning*, NeurIPS 2022.
- [ULB/Worldline credit-card fraud](https://www.openml.org/d/1597): A. Dal Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi, *Calibrating Probability with Undersampling for Unbalanced Classification*, IEEE SSCI 2015, [doi:10.1109/SSCI.2015.33](https://doi.org/10.1109/SSCI.2015.33).

The optional baselines use scikit-learn, CatBoost, XGBoost, LightGBM and the RTDL
FT-Transformer package. Dependencies and datasets retain their own terms. The
included project license does not apply to third-party packages or public data.
