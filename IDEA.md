# Split-Align Rectified LpJEPA for ImageNet-1K

## Assumption and scope

The attached Rectified LpJEPA paper and the public repo are centered on the ImageNet-100 recipe, not a released ImageNet-1K config. This document therefore defines an **ImageNet-1K adaptation** that keeps the repo's overall training style the same—ResNet-50 student, 3-layer MLP-style projector, two augmented views, RDMReg, LARS, warmup+cosine, mixed precision—and adds a **teacher-aligned split latent** on top.

The core idea is:

- split the student projector latent into a **compatible branch** `z_c` and a **free branch** `z_f`;
- align only `z_c` to a frozen large pretrained teacher;
- regularize only `z_f` with the existing Rectified LpJEPA machinery;
- explicitly decorrelate `z_c` and `z_f` so the free branch does not just copy the teacher-aligned branch.

This is meant to be implemented directly on top of the current repo structure:

- `solo/methods/rectified_lpjepa.py`
- `solo/losses/rectified_lpjepa.py`
- `scripts/pretrain/imagenet-100/rectified_lpjepa_imagenet.yaml`

The new method name should be:

`split_align_rectified_lpjepa`

---

## 1. Method summary

### Name

**Split-Align Rectified LpJEPA (SA-RLpJEPA)**

### Goal

Learn a student latent

$$
z_s = [z_c, z_f]
$$

where:

- `z_c` is a **small compatibility subspace** trained to match a frozen large teacher;
- `z_f` is a **free sparse subspace** trained with Rectified Distribution Matching Regularization (RDMReg).

The method should preserve the repo's good sparsity behavior while injecting teacher semantics into only a controlled subset of the student representation.

### Why this is a good fit for this repo

The repo already has:

- a two-view JEPA-style invariance loss,
- RDMReg / sliced Wasserstein distribution matching,
- projection utilities,
- sparsity logging,
- optional projector classifier.

So the minimal conceptual extension is **not** to replace Rectified LpJEPA, but to **factor its projector output into two branches** and add a frozen teacher path.

---

## 2. Concrete student and teacher architecture

## 2.1 Student

Keep the current student backbone unchanged for the first paper run:

- `backbone = resnet50`
- same two-view augmentation pipeline as the current repo / paper
- same encoder forward path from `BaseMethod`

Replace the current single projector with a **shared trunk + two heads**:

### Shared projector trunk

```text
feats -> Linear(features_dim, 2048)
      -> BatchNorm1d(2048)
      -> ReLU
      -> Linear(2048, 2048)
      -> BatchNorm1d(2048)
      -> ReLU
      -> h
```

### Compatible head

```text
h -> Linear(2048, D_c) -> z_c
```

- no final ReLU
- signed output is allowed
- default `D_c = 512`

### Free head

```text
h -> Linear(2048, D_f) -> ReLU -> z_f
```

- explicit non-negativity preserved here
- default `D_f = 1536`
- require `D_c + D_f = 2048`

### Final student latent used for probing / online classifier

```python
z = torch.cat([z_c, z_f], dim=1)
```

Use the concatenated `z` for the optional online projector classifier so that the reported projector probe measures the full student representation.

## 2.2 Teacher

Add a frozen teacher wrapper. Do **not** hard-code a single teacher family into the method. Support:

1. a checkpointed large JEPA / SSL model,
2. a `timm` feature extractor,
3. a plain custom PyTorch module loaded from path.

For the first implementation, the teacher interface only needs to return **one global feature vector per image**:

```python
t = teacher(images)  # shape [B, D_t]
```

Recommended default experimental setup:

- frozen large teacher,
- global pooled feature,
- L2 normalization before alignment,
- no teacher gradients,
- `eval()` mode always.

## 2.3 Alignment head

Map the small compatible branch into teacher feature space:

```text
z_c -> LayerNorm(D_c) -> Linear(D_c, D_t, bias=False) -> a
```

where:

- `a` is the teacher-aligned student vector,
- `D_t` is auto-detected from the teacher,
- the linear layer weight is named `align_head.weight`.

Default:

- `D_c = 512`
- `D_t = teacher output dim`

---

## 3. Forward pass

For two student views `x1, x2`:

```python
f1 = backbone(x1)
f2 = backbone(x2)

h1 = shared_projector(f1)
h2 = shared_projector(f2)

z_c1 = compatible_head(h1)
z_c2 = compatible_head(h2)

z_f1 = free_head(h1)   # includes final ReLU
z_f2 = free_head(h2)

z1 = cat([z_c1, z_f1], dim=1)
z2 = cat([z_c2, z_f2], dim=1)

a1 = normalize(align_head(layernorm(z_c1)), dim=1)
a2 = normalize(align_head(layernorm(z_c2)), dim=1)

with torch.no_grad():
    t1 = normalize(teacher(x1), dim=1)
    t2 = normalize(teacher(x2), dim=1)
    t_bar = normalize((t1 + t2) / 2, dim=1)
```

Important detail: use the **view-averaged teacher target** `t_bar` by default. That makes the teacher anchor less crop-specific than using `t1` and `t2` independently.

---

## 4. Concrete loss function

Let the batch size be `B`.

## 4.1 Invariance loss

Use the same JEPA-style view consistency, but apply it separately to the two branches:

$$
L_{\text{inv}} = \frac{1}{2}\left[\operatorname{MSE}(z_c^1, z_c^2) + \operatorname{MSE}(z_f^1, z_f^2)\right].
$$

This keeps the factorization explicit and prevents one branch from dominating the other numerically.

## 4.2 Free-branch RDMReg loss

Apply existing Rectified LpJEPA regularization **only** to `z_f`:

$$
L_{\text{free-rdm}} = \operatorname{RDMReg}(z_f^1, z_f^2; \mu, p, \sigma).
$$

Implementation note:

- directly reuse `rdmreg_loss(...)` from `solo/losses/rectified_lpjepa.py`
- target distribution must stay `rectified_lp_distribution`
- default: `p = 1.0`, `mu = -1.0`, `mode_of_sigma = sigma_GN`

Rationale for the default:

- `p = 1.0` keeps the strong Rectified Laplace prior that worked well in the paper,
- `mu = -1.0` gives meaningful sparsity without jumping immediately into the extreme `mu = -3` regime.

## 4.3 Instance alignment loss

Align each compatible student view to the same view-averaged teacher target:

$$
L_{\text{align-inst}} = \frac{1}{2}\left[
\big(1 - \cos(a_1, t_{\text{bar}})\big) +
\big(1 - \cos(a_2, t_{\text{bar}})\big)
\right].
$$

Use batch mean over samples.

## 4.4 Relation alignment loss

Match the within-batch geometry of the compatible branch to the teacher.

Define:

$$
\bar a = \operatorname{normalize}\left(\frac{a_1 + a_2}{2}, \text{dim}=1\right),
\qquad
G_s = \bar a\bar a^\top - I,
$$

$$
G_t = t_{\text{bar}} t_{\text{bar}}^\top - I.
$$

Then:

$$
L_{\text{align-rel}} = \frac{1}{B(B-1)} \lVert G_s - G_t \rVert_F^2.
$$

Use `diag = 0` by subtracting the identity.

Why keep this term:

- direct cosine alignment makes `z_c` match teacher instances,
- relation loss makes `z_c` match teacher **geometry** and is less brittle when `D_c << D_t`.

## 4.5 Branch decorrelation loss

Prevent `z_f` from degenerating into a copy of `z_c`.

For each view, center and standardize features over the batch:

$$
\tilde Z_c = \frac{Z_c - \mu(Z_c)}{\sigma(Z_c) + \epsilon},
\qquad
\tilde Z_f = \frac{Z_f - \mu(Z_f)}{\sigma(Z_f) + \epsilon}.
$$

Then compute cross-covariance:

$$
C_{cf} = \frac{1}{B-1} \tilde Z_c^\top \tilde Z_f.
$$

Loss:

$$
L_{\text{decouple}} = \frac{1}{2}
\left[
\frac{\lVert C_{cf}^{(1)} \rVert_F^2}{D_c D_f}
+
\frac{\lVert C_{cf}^{(2)} \rVert_F^2}{D_c D_f}
\right].
$$

This is the key term that makes the split meaningful.

## 4.6 Compatible-branch variance floor

Teacher alignment alone can still collapse `z_c` onto too few student directions. Add a VICReg-style std floor only on the compatible branch:

$$
L_{\text{c-var}} = \frac{1}{2}
\sum_{v \in \{1,2\}}
\frac{1}{D_c}
\sum_{j=1}^{D_c}
\max\big(0, \gamma - \operatorname{Std}(Z_{c,j}^{(v)})\big).
$$

Default:

- `gamma = 1.0`

## 4.7 Alignment-head orthogonality penalty

Since `D_c < D_t`, encourage the alignment head to behave like a near-isometric embedding of the small compatible space into teacher space.

If `W = align_head.weight` with shape `[D_t, D_c]`, add:

$$
L_{\text{orth}} = \frac{\lVert W^\top W - I_{D_c} \rVert_F^2}{D_c^2}.
$$

This makes `z_c` preserve geometry better before entering teacher space.

## 4.8 Total loss

Use:

$$
L =
\lambda_{\text{inv}} L_{\text{inv}}
+
\lambda_{\text{rdm}} L_{\text{free-rdm}}
+
\omega(e)\Big[
\lambda_{\text{inst}} L_{\text{align-inst}}
+
\lambda_{\text{rel}} L_{\text{align-rel}}
\Big]
+
\lambda_{\text{dec}} L_{\text{decouple}}
+
\lambda_{\text{var}} L_{\text{c-var}}
+
\lambda_{\text{orth}} L_{\text{orth}}.
$$

### Default weights

```text
lambda_inv   = 25.0
lambda_rdm   = 125.0
lambda_inst  = 10.0
lambda_rel   = 1.0
lambda_dec   = 0.05
lambda_var   = 5.0
lambda_orth  = 1e-3
```

These are the starting values, not sacred constants.

---

## 5. Alignment schedule

Use strong teacher alignment early, then decay it late so the student can use `z_f` more freely.

Let `E` be total epochs and `e` the current epoch.

$$
\omega(e) =
\begin{cases}
1, & e < 0.6E \\
\omega_{\min} + (1-\omega_{\min})\cdot \frac{1}{2}\left(1 + \cos\left(\pi\frac{e-0.6E}{0.4E}\right)\right), & e \ge 0.6E
\end{cases}
$$

Default:

- `omega_min = 0.2`

So alignment stays on for the whole run, but becomes weaker in the final 40%.

---

## 6. Default ImageNet-1K training recipe

Keep the paper / repo recipe as intact as possible.

## 6.1 Data

Use:

```yaml
data:
  dataset: imagenet
  train_path: /path/to/imagenet/train
  val_path: /path/to/imagenet/val
  format: image_folder
  preload: false
  num_workers: 16
```

Important: set `preload: false` for ImageNet-1K.

## 6.2 Augmentations

Keep the current repo / paper augmentation stack:

- random resized crop, scale `[0.2, 1.0]`, size `224`
- horizontal flip `0.5`
- color jitter `0.8`, brightness `0.4`, contrast `0.4`, saturation `0.2`, hue `0.1`
- grayscale `0.2`
- Gaussian blur `0.5`
- solarization `0.1`
- `num_crops: 2`

Do **not** change augmentations in the first version.

## 6.3 Optimizer / scheduler

Keep the same style as the repo:

- optimizer: `LARS`
- scheduler: `warmup_cosine`
- weight decay: `1e-4`
- mixed precision: `16-mixed`
- sync batch norm if using DDP

Recommended first-pass policy:

- keep the repo's per-process batch-size convention,
- scale LR linearly with effective global batch only after the method is stable.

Safer first pass:

```yaml
optimizer:
  name: lars
  batch_size: 128
  lr: 0.165
  classifier_lr: 0.055
  weight_decay: 1e-4
```

If you run multi-GPU with a larger effective batch, scale carefully after a pilot run.

## 6.4 Epochs

Use:

- `max_epochs: 1000`

for the main paper-quality run.

For debugging:

- `100 epochs` smoke test,
- `300 epochs` intermediate ablation sweep,
- `1000 epochs` final runs.

---

## 7. Files to add or modify

## 7.1 New loss file

Add:

```text
solo/losses/split_align_rectified_lpjepa.py
```

It should contain:

- `instance_alignment_loss(...)`
- `relation_alignment_loss(...)`
- `branch_decorrelation_loss(...)`
- `compatible_variance_loss(...)`
- `orthogonality_loss(...)`
- `split_align_rectified_lp_jepa_loss(...)`

Implementation note:

- import and reuse `rdmreg_loss`, `determine_sigma_for_lp_dist`, `choose_sigma_for_unit_var` from `solo/losses/rectified_lpjepa.py`

## 7.2 New method file

Add:

```text
solo/methods/split_align_rectified_lpjepa.py
```

This class should either:

- subclass `RectifiedLpJEPA`, or
- subclass `BaseMethod` and reuse `Projections` from `solo/methods/rectified_lpjepa.py`.

Recommended path: **subclass `RectifiedLpJEPA`** and override only what changes.

The class should:

- build shared projector trunk,
- build `compatible_head`, `free_head`, `align_head`,
- load and freeze the teacher,
- expose `z_c`, `z_f`, and concatenated `z`,
- compute the new total loss in `training_step`,
- keep the online projector classifier path unchanged.

## 7.3 Teacher utility

Add:

```text
solo/utils/teacher_wrapper.py
```

This wrapper should:

- load teacher from config,
- normalize inputs if teacher needs a different normalization,
- return a single `[B, D_t]` tensor,
- stay in `eval()` and `no_grad()` mode,
- optionally autocast to fp16/bf16 for cheaper inference.

## 7.4 Method registry

Modify:

```text
solo/methods/__init__.py
```

Register:

```python
"split_align_rectified_lpjepa": SplitAlignRectifiedLpJEPA
```

## 7.5 Config

Add:

```text
scripts/pretrain/imagenet/split_align_rectified_lpjepa_imagenet.yaml
```

If you prefer consistency with the existing directory style, put it under a new ImageNet-1K folder and keep the old ImageNet-100 config untouched.

---

## 8. Recommended config skeleton

```yaml
name: "split-align-rectified-lpjepa-imagenet1k"
method: "split_align_rectified_lpjepa"
backbone:
  name: "resnet50"

method_kwargs:
  proj_hidden_dim: 2048
  proj_output_dim: 2048
  compatible_dim: 512
  free_dim: 1536

  target_distribution: "rectified_lp_distribution"
  lp_norm_parameter: 1.0
  mean_shift_value: -1.0
  mode_of_sigma: "sigma_GN"

  invariance_loss_weight: 25.0
  rdm_reg_loss_weight: 125.0
  align_instance_weight: 10.0
  align_relation_weight: 1.0
  branch_decorr_weight: 0.05
  compatible_var_weight: 5.0
  compatible_var_gamma: 1.0
  orth_weight: 1.0e-3

  num_projections: 8192
  projection_vectors_type: "random"

  align_schedule: "late_cosine_decay"
  align_decay_start_pct: 0.6
  align_decay_end_weight: 0.2

  teacher_source: "checkpoint"
  teacher_ckpt_path: "/path/to/teacher.ckpt"
  teacher_feature_key: "feats"
  teacher_pooling: "global"
  teacher_normalize: true
  teacher_target_mode: "view_average"

  add_projector_classifier: true
  logging_interval: 50

data:
  dataset: imagenet
  train_path: "/path/to/imagenet/train"
  val_path: "/path/to/imagenet/val"
  preload: false
  format: image_folder
  num_workers: 16

optimizer:
  name: lars
  batch_size: 128
  lr: 0.165
  classifier_lr: 0.055
  weight_decay: 1e-4
  kwargs:
    clip_lr: true
    eta: 0.02
    exclude_bias_n_norm: true

scheduler:
  name: warmup_cosine

max_epochs: 1000
sync_batchnorm: true
accelerator: gpu
strategy: ddp
precision: 16-mixed
```

---

## 9. Pseudocode for `training_step`

```python
def training_step(self, batch, batch_idx):
    out = super().training_step(batch, batch_idx)
    class_loss = out["loss"]

    x1, x2 = out["X"]               # or unpack original views from batch pipeline
    z_c1, z_c2 = out["z_c"]
    z_f1, z_f2 = out["z_f"]
    z1, z2     = out["z"]

    proj_vecs = Projections.get_projection_vectors(
        z_f1, z_f2,
        self.num_projections,
        self.projection_vectors_type,
        self.free_dim,
    )

    with torch.no_grad():
        t1 = F.normalize(self.teacher(x1), dim=1)
        t2 = F.normalize(self.teacher(x2), dim=1)
        t_bar = F.normalize((t1 + t2) / 2, dim=1)

    a1 = F.normalize(self.align_head(self.compat_ln(z_c1)), dim=1)
    a2 = F.normalize(self.align_head(self.compat_ln(z_c2)), dim=1)

    loss, logs = split_align_rectified_lp_jepa_loss(
        z_c1, z_c2,
        z_f1, z_f2,
        a1, a2,
        t_bar,
        self.align_head.weight,
        proj_vecs,
        epoch=self.current_epoch,
        max_epochs=self.trainer.max_epochs,
        ...
    )

    projector_class_loss = 0.0
    if self.projector_classifier is not None:
        projector_class_loss = projector_probe_loss(z1, z2, targets)

    total = loss + class_loss + projector_class_loss
    self.log_dict(logs, on_epoch=True, sync_dist=True)
    return total
```

Key implementation detail: generate projection vectors from `z_f` only, because only the free branch is distribution-matched.

---

## 10. Logging and diagnostics

Log all current baseline metrics, plus these new ones:

### Core loss terms

- `train_split_align_total_loss`
- `train_invariance_loss`
- `train_free_rdm_loss`
- `train_align_instance_loss`
- `train_align_relation_loss`
- `train_branch_decorr_loss`
- `train_compatible_var_loss`
- `train_orth_loss`
- `train_align_schedule_weight`

### Alignment health

- `train_teacher_student_cosine`
- `train_teacher_relation_mse`
- `train_align_head_fro_norm`
- `train_align_head_orth_gap = ||W^T W - I||_F`

### Sparsity

Log separately for `z_f` and full `z`:

- `train_l0_sparsity_metric_zf`
- `train_l1_sparsity_metric_zf`
- `train_l0_sparsity_metric_full`
- `train_l1_sparsity_metric_full`

### Collapse / redundancy checks

- `train_variance_loss_zc`
- `train_covariance_loss_zc`
- `train_crosscov_zc_zf`
- optional: `train_nhsic_zc_zf`

### Online classifier

Keep existing:

- `train_proj_loss`
- `train_proj_acc1`

---

## 11. Reduction tests that should hold

These are extremely useful for debugging.

### Test A: baseline recovery

Set:

- `compatible_dim = 0`
- `free_dim = 2048`
- all teacher losses off

Then the method should reduce to the current Rectified LpJEPA baseline.

### Test B: pure teacher-aligned student

Set:

- `free_dim = 0`
- `compatible_dim = 2048`
- `lambda_rdm = 0`

Then the method becomes a teacher-aligned JEPA without sparse free branch.

### Test C: split but no teacher

Set:

- `lambda_inst = 0`
- `lambda_rel = 0`

Then `z_c` and `z_f` should still train without numerical issues, and `z_f` should keep the expected sparsity pattern.

### Test D: orthogonality off

Set:

- `lambda_orth = 0`

This should still train. If not, the issue is not the orthogonality term.

---

## 12. Main ablation grid

Run the ablations in this order.

## 12.1 Minimal paper grid

| ID | Variant | Change from full method | Purpose |
|---|---|---|---|
| A0 | Rectified LpJEPA baseline | no teacher, no split | base repo baseline |
| A1 | Full-latent alignment baseline | align all 2048 dims, no split | test if splitting matters |
| A2 | Split, no decouple | `lambda_dec = 0` | test whether branches collapse into each other |
| A3 | Split, no relation | `lambda_rel = 0` | test geometry matching vs instance-only |
| A4 | Split, no late decay | constant alignment weight | test if late relaxation matters |
| A5 | Full SA-RLpJEPA | default settings | main result |

## 12.2 Split-size ablation

Hold everything else fixed.

| ID | `D_c` | `D_f` |
|---|---:|---:|
| S1 | 256 | 1792 |
| S2 | 512 | 1536 |
| S3 | 1024 | 1024 |

Expectation:

- too-small `D_c` underfits teacher semantics,
- too-large `D_c` reduces the value of the free sparse branch.

## 12.3 Free-branch sparsity ablation

Use `sigma_GN` throughout.

| ID | `p` | `mu` |
|---|---:|---:|
| F1 | 1.0 | 0.0 |
| F2 | 1.0 | -1.0 |
| F3 | 1.0 | -2.0 |
| F4 | 2.0 | 0.0 |
| F5 | 2.0 | -1.0 |
| F6 | 2.0 | -2.0 |

Recommended default for the full method: **F2**.

## 12.4 Projection-vector ablation

| ID | Projection type | Purpose |
|---|---|---|
| P1 | `random` | lowest engineering risk |
| P2 | `torch_svd_bottom_half_eigen_and_random` | test faster convergence on `z_f` |
| P3 | `torch_svd_and_random` | stronger eigendirection matching |

Start with `random`. Move to `bottom_half_eigen_and_random` only after the full method is stable.

## 12.5 Teacher-target ablation

| ID | Target mode |
|---|---|
| T1 | per-view teacher target |
| T2 | view-averaged teacher target |

Recommended default: **T2**.

---

## 13. What to report

For every main run, report:

1. **ImageNet-1K linear probe top-1** on
   - encoder features,
   - full projector features `z`.
2. **Sparsity**
   - `l0`, `l1` on `z_f`,
   - `l0`, `l1` on full `z`.
3. **Teacher agreement**
   - mean cosine between `a` and teacher target,
   - relation loss / Gram matching error.
4. **Branch separation**
   - cross-covariance between `z_c` and `z_f`.
5. **Training cost**
   - images/sec,
   - peak GPU memory,
   - wall clock overhead vs baseline Rectified LpJEPA.

Nice-to-have:

- k-NN retrieval on ImageNet-1K val,
- few-shot transfer on the same downstream suite used by the paper,
- nearest-neighbor visualizations comparing `z_c` vs `z_f` vs `z`.

---

## 14. Practical implementation notes

### 14.1 Keep the teacher simple

The teacher path should be frozen, single-vector, and global. Do not add token-level distillation in v1.

### 14.2 Do not put RDMReg on `z_c`

That defeats the purpose of having a teacher-compatible subspace.

### 14.3 Do not ReLU the concatenated latent

Only `z_f` should be rectified. `z_c` should remain signed.

### 14.4 Teacher forward should be cheap

Use:

- `torch.no_grad()`
- `autocast`
- `eval()`

and keep the teacher outside the optimizer.

### 14.5 Start with random projections

Even though the paper showed useful eigenvector-based projections, the cheapest first implementation on ImageNet-1K should keep `projection_vectors_type = random`.

---

## 15. Failure modes and fixes

### Failure mode: `z_f` stops being sparse

Likely causes:

- teacher alignment too strong,
- `D_c` too large,
- `mu` not negative enough.

Fixes:

- lower `lambda_inst`,
- start decay earlier (`0.5E` instead of `0.6E`),
- try `mu = -2`.

### Failure mode: `z_c` collapses / low rank

Likely causes:

- compatible branch too small,
- no variance floor,
- relation loss off.

Fixes:

- increase `lambda_var`,
- turn on relation loss,
- move from `D_c = 256` to `512`.

### Failure mode: `z_c` and `z_f` become redundant

Fixes:

- raise `lambda_dec`,
- lower `D_c`,
- decay teacher alignment slightly earlier.

### Failure mode: relation loss unstable

Fixes:

- compute on view-averaged normalized features,
- remove diagonal before Frobenius norm,
- warm up relation loss for first 10 epochs if needed.

---

## 16. Recommended execution plan

### Phase 1: smoke test

- ImageNet-100 or a 100-class ImageNet-1K subset
- `100 epochs`
- `D_c=512`, `D_f=1536`
- `p=1`, `mu=-1`
- random projections
- teacher instance loss on, relation loss off

Goal: confirm the split implementation is numerically sound.

### Phase 2: main ImageNet-1K sweep

Run:

- A0, A1, A5
- split-size ablation S1–S3
- sparsity ablation F1–F6

### Phase 3: cleanup ablations

Run:

- A2, A3, A4
- projection ablation P1–P3
- teacher-target ablation T1–T2

---

## 17. Bottom line

The full method to implement is:

- **teacher-aligned small compatible branch** `z_c`,
- **sparse rectified free branch** `z_f`,
- **RDMReg only on `z_f`**,
- **teacher instance + relation alignment only on `z_c`**,
- **explicit cross-branch decorrelation**,
- **late decay of alignment strength**.

If this works, the strongest paper claim is not just "teacher distillation helps," but:

> a JEPA student should not be forced to devote its entire latent to the teacher; the best trade-off comes from reserving a small teacher-compatible subspace and a larger sparse free subspace.

That claim is novel, cleanly testable, and well matched to the current Rectified LpJEPA codebase.
