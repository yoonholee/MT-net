# MT-net

Code accompanying the paper [Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace (Yoonho Lee and Seungjin Choi, ICML 2018)](https://arxiv.org/abs/1801.05558).
It includes code for running the experiments in the paper (few-shot sine wave regression, Omniglot and miniImagenet few-shot classification).

### Data
For the Omniglot and MiniImagenet data, see the usage instructions in `data/omniglot_resized/resize_images.py` and `data/miniImagenet/proc_images.py` respectively.

### Usage
To run the code, see the usage instructions at the top of `main.py`.

For MT-nets, set `use_T`, `use_M`, `share_M` to `True`.

For T-nets, set `use_T` to `True` and `use_M` to `False`.

---

This codebase is based on the [MAML repository](https://github.com/cbfinn/maml).
