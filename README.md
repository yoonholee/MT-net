# MT-net

Code accompanying the paper [Gradient-Based Meta-Learning with Learned Layerwise Metric and Subspace (Yoonho Lee and Seungjin Choi, ICML 2018)](https://arxiv.org/abs/1801.05558).
It includes code for running the experiments in the paper (few-shot sine wave regression, Omniglot and miniImagenet few-shot classification).

## Abstract
<img src="https://github.com/yoonholee/MT-net/blob/master/mtnet-subspace.png" height="250" /><img src="https://github.com/yoonholee/MT-net/blob/master/mtnet-fig.png" height="250" />

Gradient-based meta-learning methods leverage gradient descent to learn the commonalities among various tasks. While previous such methods have been successful in meta-learning tasks, they resort to simple gradient descent during meta-testing. Our primary contribution is the **MT-net**, which enables the meta-learner to learn on each layer's activation space a subspace that the task-specific learner performs gradient descent on. Additionally, a task-specific learner of an {\em MT-net} performs gradient descent with respect to a meta-learned distance metric, which warps the activation space to be more sensitive to task identity. We demonstrate that the dimension of this learned subspace reflects the complexity of the task-specific learner's adaptation task, and also that our model is less sensitive to the choice of initial learning rates than previous gradient-based meta-learning methods. Our method achieves state-of-the-art or comparable performance on few-shot classification and regression tasks.

### Data
For the Omniglot and MiniImagenet data, see the usage instructions in `data/omniglot_resized/resize_images.py` and `data/miniImagenet/proc_images.py` respectively.

### Usage
To run the code, see the usage instructions at the top of `main.py`.

For MT-nets, set `use_T`, `use_M`, `share_M` to `True`.

For T-nets, set `use_T` to `True` and `use_M` to `False`.

## Reference

If you found the provided code useful, please cite our work.

```
@inproceedings{lee2018gradient,
  title={Gradient-based meta-learning with learned layerwise metric and subspace},
  author={Lee, Yoonho and Choi, Seungjin},
  booktitle={International Conference on Machine Learning},
  pages={2933--2942},
  year={2018}
}
```

---

This codebase is based on the repository for [MAML](https://github.com/cbfinn/maml).
