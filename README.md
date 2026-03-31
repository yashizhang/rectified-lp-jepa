# Rectified LpJEPA

<p align="left">
  <a href="https://arxiv.org/abs/2602.01456">
    <img src="https://img.shields.io/badge/arXiv-2602.01456-b31b1b.svg" />
  </a>
  <a href="https://yilunkuang.github.io/blog/2026/rectified-lp-jepa/">
    <img src="https://img.shields.io/badge/Blog-Interactive%20Post-4F6DB8.svg" />
  </a>
</p>


This repository contains the code for **Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations** ([arXiv](https://arxiv.org/abs/2602.01456), [blog post](https://yilunkuang.github.io/blog/2026/rectified-lp-jepa/)).


<!-- --- -->

## Overview

**Rectified LpJEPA** is a Joint-Embedding Predictive Architecture (JEPA) that learns sparse, non-negative, maximum-entropy representations by aligning features to a **Rectified Generalized Gaussian** distribution using **Rectified Distribution Matching Regularization (RDMReg)**. 

By aligning features to a Rectified Generalized Gaussian distribution, RDMReg provides explicit control over the **expected $\ell_0$ sparsity** while preserving the **maximum-entropy** property, up to rescaling, under **expected $\ell_p$-norm constraints**. This results in <u>favorable sparsity-performance trade-offs</u>, enabling <u>highly sparse representations</u> while <u>preserving task-relevant information</u>.

<div align="center">
  <img src="assets/fig1_rec_lp_jepa.png" width="100%"/>
</div>



## Installation

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Core Rectified LpJEPA Code

The core Rectified LpJEPA implementations can be found in `solo/losses/rectified_lpjepa.py` and `solo/methods/rectified_lpjepa.py`.


## Rectified LpJEPA Pretraining over CIFAR-100

See [MINIMAL.md](MINIMAL.md) for a standalone implementation of Rectified LpJEPA training over CIFAR-100.

## Rectified LpJEPA Pretraining over ImageNet-100

### Default Script

To pretrain a Rectified LpJEPA model over ImageNet-100 with a ResNet-50 backbone, run

```bash
python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=rectified_lpjepa_imagenet.yaml \
  ++wandb.entity=<ENTITY> \
  ++wandb.project=<PROJECT> \
  ++wandb.enabled=true # set to false for debugging
```

By default, `rectified_lpjepa_imagenet.yaml` sets the target distribution to be the Rectified Generalized Gaussian distribution with location parameter $\mu=0$, scale parameter $\sigma_{\text{GN}}=1/\sqrt{2}$, and $\ell_p$ parameter $p=1$. The target distribution in this case is also called the Rectified Laplace distribution, which is our default choice. 

To use another target distribution within the Rectified Generalized Gaussian family, consider running the following script:

### Script with Varying Target Distributions

```bash
python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=rectified_lpjepa_imagenet.yaml \
  ++method_kwargs.lp_norm_parameter=2.0 \
  ++method_kwargs.mean_shift_value=-1.0 \
  ++wandb.entity=<ENTITY> \
  ++wandb.project=<PROJECT> \
  ++wandb.enabled=true # set to false for debugging
```

where we update the target distribution to be the Rectified Gaussian distribution with location parameter $\mu=-1$, scale parameter $\sigma_{\text{GN}}$, and $\ell_p$ parameter $p=2$. By definition, $\sigma_{\text{GN}}=\Gamma(1/p)^{1/2}/(p^{1/p}\cdot\Gamma(3/p)^{1/2})$, where $\Gamma(\cdot)$ is the Gamma function. When $p=2$, $\sigma_{\text{GN}}=1.0$.


### Varying Both Target Distributions and Other Hyperparameters

Common tunable hyperparameters include `++method_kwargs.invariance_loss_weight`, `++method_kwarsg.rdm_reg_loss_weight`, which are hyperparameters for the invariance and the RDMReg loss respectively. The learning rates `++optimizer.lr` and `++optimizer.classifier_lr` can also be changed accordingly. One example script to include all these arguments is

```bash
python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=rectified_lpjepa_imagenet.yaml \
  ++method_kwargs.lp_norm_parameter=2.0 \
  ++method_kwargs.mean_shift_value=-1.0 \
  ++method_kwargs.invariance_loss_weight=25.0 \
  ++method_kwarsg.rdm_reg_loss_weight=125.0 \
  ++optimizer.lr=0.165 \
  ++optimizer.lr=0.055 \
  ++wandb.entity=<ENTITY> \
  ++wandb.project=<PROJECT> \
  ++wandb.enabled=true # set to false for debugging
```


<!-- For a complete SLURM-ready reproduction script, refer to:
`sbatch_scripts/imagenet_torch_run.sh`

```bash
python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=rectified_lpjepa_imagenet.yaml \
  ++method="rectified_lpjepa" \
  ++method_kwargs.mean_shift_value=-1.0
``` -->



## Split Teacher + SIGReg JEPA with DinoV2 on `z_c`

The repository also includes a **split teacher + SIGReg JEPA** path in which:

- `z_c` is distilled against a frozen **DinoV2** teacher,
- `z_f` follows the **LeJEPA/SIGReg** recipe, and
- the predictive loss is applied on the full latent `torch.cat([z_c, z_f], dim=1)`.

The default ImageNet-100 config is:

- `scripts/pretrain/imagenet-100/split_teacher_sigjepa_imagenet100.yaml`
- student backbone: `vit_base` with `patch_size=14`
- teacher backend: `hf_dinov2`
- teacher model id: `facebook/dinov2-large`
- teacher pooling: `cls`

### Online DinoV2 teacher

```bash
python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  ++wandb.entity=<ENTITY> \
  ++wandb.project=<PROJECT> \
  ++wandb.enabled=true
```

### Prefetch DinoV2 embeddings to storage

```bash
python3 main_prefetch_teacher.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=dinov2_sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.cache_dir=/path/to/storage/teacher_prefetch_cache/dinov2_sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.num_epochs=1 \
  ++method_kwargs.teacher_prefetch.dtype=float16
```

Then point training at the same cache directory with `++method_kwargs.teacher_prefetch.enabled=true`. See [`commands.md`](commands.md) for complete examples.

## Acknowledgements

This codebase is built upon the [solo-learn](https://github.com/vturrisi/solo-learn) framework.
We thank the solo-learn authors for releasing their code under the MIT license.

## Citation
Please cite our work if you find it helpful:
```bibtex
@misc{kuang2026rectifiedlpjepajointembeddingpredictive,
      title={Rectified LpJEPA: Joint-Embedding Predictive Architectures with Sparse and Maximum-Entropy Representations}, 
      author={Yilun Kuang and Yash Dagade and Tim G. J. Rudner and Randall Balestriero and Yann LeCun},
      year={2026},
      eprint={2602.01456},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.01456}, 
}
```
