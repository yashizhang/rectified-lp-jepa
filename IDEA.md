# Barebones Split-Teacher SIGReg JEPA for ImageNet-100

## 0. Purpose

This document is a **coding spec** for a minimal first implementation on top of the `rectified-lp-jepa` repo.

The goal is to keep the method as simple as possible:

1. **teacher loss on `z_c` only**,
2. **regularization loss on `z_f` only**,
3. **predictive loss on `torch.cat([z_c, z_f], dim=1)`**.

This version is intentionally minimal and should be preferred over more ambitious variants for the first pass.

---

## 1. Hard constraints for v0

### Use the same ImageNet-100 training setting as the repo/paper

Keep the released ImageNet-100 setup:

- dataset: `imagenet100`
- backbone: `resnet50`
- projector: 3-layer MLP, hidden dim `2048`, output dim `2048`
- two augmented crops
- crop size `224`
- random resized crop scale `[0.2, 1.0]`
- horizontal flip `0.5`
- color jitter `0.8` with brightness `0.4`, contrast `0.4`, saturation `0.2`, hue `0.1`
- grayscale `0.2`
- Gaussian blur `0.5`
- solarization `0.1`
- optimizer: `AdamW`
- batch size `128`
- learning rate `0.0825`
- classifier learning rate `0.0275`
- weight decay `1e-4`
- scheduler: warmup + cosine
- `max_epochs = 1000`
- `precision = 16-mixed`

### Do not use Rectified LpJEPA ideas in the proposed method

For the proposed method, **do not** use:

- RDMReg,
- rectified target distributions,
- final ReLU on the projector output,
- sparsity-specific rectified losses.

The repo is only the **codebase to build on**, not the source of the new method design.

### Do not add extra losses in v0

Do **not** add these in the first implementation:

- relation distillation,
- branch decorrelation,
- orthogonality penalties,
- predictor heads,
- EMA teacher,
- late-stage schedules,
- extra branch-specific MLPs beyond what is explicitly listed below.

If the minimal version works, these can be added later.

---

## 2. Proposed method name

Use:

`split_teacher_sigjepa`

This name reflects exactly what v0 is:

- split latent,
- frozen teacher supervision,
- SIGReg on the free branch,
- JEPA-style predictive loss.

---

## 3. High-level idea

Let the student projector output be

```python
z = projector(feats)   # shape [B, 2048]
```

Split it into two contiguous chunks:

```python
z_c = z[:, :compatible_dim]
z_f = z[:, compatible_dim:]
```

where:

- `z_c` = **compatible** subspace, aligned to a frozen teacher,
- `z_f` = **free** subspace, regularized only by `SIGReg`.

Then train with:

- predictive loss on the **full latent** `z = cat([z_c, z_f])`,
- teacher loss only on `z_c`,
- SIGReg only on `z_f`.

This is the minimal implementation the user asked for.

---

## 4. Architecture

## 4.1 Student backbone

Keep the current repo choice for the main run:

- `backbone = resnet50`

No change to the encoder path.

## 4.2 Student projector

Use a standard 3-layer MLP, matching the repo dimensions, but **without** a final ReLU:

```python
projector = nn.Sequential(
    nn.Linear(features_dim, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(),
    nn.Linear(2048, 2048),
)
```

Important:

- output dimension stays `2048`,
- output is **signed**, not rectified.

## 4.3 Latent split

Default split:

- `compatible_dim = 512`
- `free_dim = 1536`
- require `compatible_dim + free_dim == proj_output_dim`

Split by slicing only. Do **not** make two separate projector heads in v0.

This keeps the implementation close to the current repo and reduces code changes.

## 4.4 Teacher branch

Use a frozen teacher wrapper that returns one vector per image:

```python
t = teacher(x)  # shape [B, teacher_dim]
```

Requirements:

- `teacher.eval()` always,
- `torch.no_grad()` always,
- teacher parameters excluded from optimizer,
- output dim `teacher_dim` inferred at runtime if possible.

## 4.5 Alignment head

Map `z_c` to teacher space with a single linear layer:

```python
align_head = nn.Linear(compatible_dim, teacher_dim, bias=False)
```

If `compatible_dim == teacher_dim`, allow an optional identity mapping later, but use the linear layer by default in v0.

---

## 5. Exact forward pass

For two views `x1, x2`:

```python
out1 = super().forward(x1)
out2 = super().forward(x2)

f1 = out1["feats"]
f2 = out2["feats"]

z1 = projector(f1)
z2 = projector(f2)

z_c1 = z1[:, :compatible_dim]
z_c2 = z2[:, :compatible_dim]

z_f1 = z1[:, compatible_dim:]
z_f2 = z2[:, compatible_dim:]

a1 = F.normalize(align_head(z_c1), dim=1)
a2 = F.normalize(align_head(z_c2), dim=1)

with torch.no_grad():
    t1 = F.normalize(teacher(x1), dim=1)
    t2 = F.normalize(teacher(x2), dim=1)
```

Return all of the following from `forward`:

```python
{
    "logits": logits,
    "feats": feats,
    "z": z,
    "z_c": z_c,
    "z_f": z_f,
}
```

This keeps the `BaseMethod` interface intact.

---

## 6. Losses

## 6.1 Predictive loss on the full latent

The predictive loss is applied to the **full** student latent:

```python
z1_full = torch.cat([z_c1, z_f1], dim=1)  # same as z1
z2_full = torch.cat([z_c2, z_f2], dim=1)  # same as z2
```

### Minimal implementation

Use plain MSE between the two views:

```python
L_pred = F.mse_loss(z1_full, z2_full)
```

### Why this is okay

In LeJEPA, the prediction loss is defined on multiple views. With only two global views and no local views, the full predictive loss is proportional to pairwise MSE between the two view embeddings. So for this repo and this v0 implementation, plain MSE is the correct simple choice.

## 6.2 Teacher loss on `z_c` only

Use normalized MSE to align `z_c` to the frozen teacher:

```python
L_teacher = 0.5 * (
    F.mse_loss(a1, t1) +
    F.mse_loss(a2, t2)
)
```

where:

```python
a1 = F.normalize(align_head(z_c1), dim=1)
a2 = F.normalize(align_head(z_c2), dim=1)
t1 = F.normalize(teacher(x1), dim=1)
t2 = F.normalize(teacher(x2), dim=1)
```

This is equivalent to cosine alignment up to constants and is easy to implement.

## 6.3 Regularization loss on `z_f` only

Default regularizer:

```python
L_free = 0.5 * (
    sigreg(z_f1, global_step=self.global_step, ...) +
    sigreg(z_f2, global_step=self.global_step, ...)
)
```

This is the only regularizer on the free branch in v0.

## 6.4 Total SSL loss

Use the simplest weighted sum:

```python
L_ssl = (
    lambda_pred * L_pred
    + lambda_teacher * L_teacher
    + lambda_sigreg * L_free
)
```

Recommended default weights:

```python
lambda_pred = 1.0
lambda_teacher = 1.0
lambda_sigreg = 0.05
```

Why these defaults:

- `lambda_pred = 1.0`: keep the full latent predictive task as the anchor loss,
- `lambda_teacher = 1.0`: teacher supervision should matter from the start,
- `lambda_sigreg = 0.05`: consistent with LeJEPA-style regularization magnitude for a first pass.

### Final training loss returned by the method

Keep the repo behavior:

```python
total = L_ssl + class_loss + projector_class_loss
```

where:

- `class_loss` comes from the online encoder classifier already in `BaseMethod`,
- `projector_class_loss` is optional and uses the full `z`.

---

## 7. What is explicitly **not** in v0

Do not implement any of the following in the first pass:

```text
- no relation loss
- no teacher-view averaging
- no z_c / z_f decorrelation
- no orthogonality penalty on align_head
- no EMA teacher
- no teacher schedule
- no separate projector trunks/heads
- no RDMReg
- no final ReLU on z
```

The first version should be the smallest possible method that tests the core idea.

---

## 8. SIGReg pseudocode (LeJEPA-style)

This section should be implemented as a standalone utility so it can be reused later.

## 8.1 Interface

```python
def sigreg(
    x: torch.Tensor,
    global_step: int,
    num_slices: int = 256,
    num_points: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    ddp_sync: bool = True,
) -> torch.Tensor:
    ...
```

Input:

- `x`: shape `[N, K]`
- returns: scalar loss

## 8.2 Paper-aligned pseudocode

This is the LeJEPA Algorithm 1 logic adapted to the repo style.

```python
def sigreg(x, global_step, num_slices=256, num_points=17, t_min=-5.0, t_max=5.0):
    # x: [N, K]
    device = x.device
    dtype = x.dtype

    # 1. sample random unit directions, seeded by global_step
    g = torch.Generator(device=device)
    g.manual_seed(int(global_step))
    A = torch.randn(x.size(1), num_slices, generator=g, device=device, dtype=dtype)
    A = F.normalize(A, dim=0)                         # [K, M]

    # 2. integration points for the characteristic function test
    t = torch.linspace(t_min, t_max, num_points, device=device, dtype=dtype)   # [T]

    # 3. target characteristic function for N(0, 1)
    target_cf = torch.exp(-0.5 * t.square())         # [T]

    # 4. empirical characteristic function on projected samples
    #    projected: [N, M]
    projected = x @ A
    projected_t = projected.unsqueeze(-1) * t.view(1, 1, -1)   # [N, M, T]

    ecf = torch.exp(1j * projected_t).mean(dim=0)    # [M, T]

    # 5. DDP average if needed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(ecf, op=torch.distributed.ReduceOp.SUM)
        ecf = ecf / torch.distributed.get_world_size()
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    # 6. weighted L2 distance between empirical and target characteristic functions
    err = (ecf - target_cf.view(1, -1)).abs().square() * target_cf.view(1, -1)

    # 7. integrate over t, then average over slices
    per_slice = torch.trapz(err, t, dim=-1) * (x.size(0) * world_size)   # [M]
    return per_slice.mean()
```

## 8.3 Safer real-valued version

If complex autograd causes issues under AMP, use the equivalent real-valued form:

```python
def sigreg_real(x, global_step, num_slices=256, num_points=17, t_min=-5.0, t_max=5.0):
    device = x.device
    dtype = x.dtype

    g = torch.Generator(device=device)
    g.manual_seed(int(global_step))
    A = torch.randn(x.size(1), num_slices, generator=g, device=device, dtype=dtype)
    A = F.normalize(A, dim=0)

    t = torch.linspace(t_min, t_max, num_points, device=device, dtype=dtype)
    target_cf = torch.exp(-0.5 * t.square())

    projected = x @ A
    projected_t = projected.unsqueeze(-1) * t.view(1, 1, -1)

    ecf_real = torch.cos(projected_t).mean(dim=0)
    ecf_imag = torch.sin(projected_t).mean(dim=0)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(ecf_real, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ecf_imag, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
        ecf_real = ecf_real / world_size
        ecf_imag = ecf_imag / world_size
    else:
        world_size = 1

    err = ((ecf_real - target_cf.view(1, -1)) ** 2 + ecf_imag ** 2) * target_cf.view(1, -1)
    per_slice = torch.trapz(err, t, dim=-1) * (x.size(0) * world_size)
    return per_slice.mean()
```

### Recommendation

Start with the complex version. If AMP or complex dtype causes headaches, switch to `sigreg_real`.

---

## 9. Repo changes

## 9.1 New files

Add:

```text
solo/losses/sigreg.py
solo/losses/split_teacher_sigjepa.py
solo/utils/teacher.py
solo/methods/split_teacher_sigjepa.py
scripts/pretrain/imagenet-100/split_teacher_sigjepa_imagenet100.yaml
```

## 9.2 File responsibilities

### `solo/losses/sigreg.py`

Contains only SIGReg-related helpers:

- `sample_unit_sphere(...)`
- `sigreg(...)`
- `sigreg_real(...)`

Keep this file independent from the rest of the method.

### `solo/losses/split_teacher_sigjepa.py`

Contains only simple pure loss functions:

- `predictive_loss(z1, z2)`
- `teacher_alignment_loss(a1, a2, t1, t2)`
- `free_regularization_loss(z_f1, z_f2, ...)`
- `split_teacher_sigjepa_loss(...)`

Make sure these are plain functions, not methods.

### `solo/utils/teacher.py`

Contains a tiny frozen teacher wrapper:

- load teacher from checkpoint or timm,
- expose `forward(x) -> [B, teacher_dim]`,
- stay in `eval()` and `no_grad()`.

### `solo/methods/split_teacher_sigjepa.py`

Implements the actual method class.

### `scripts/pretrain/imagenet-100/split_teacher_sigjepa_imagenet100.yaml`

Contains the v0 training config.

---

## 10. Method implementation details

## 10.1 Method class skeleton

Implement a new method class:

```python
class SplitTeacherSIGJEPA(BaseMethod):
    ...
```

Do **not** subclass `RectifiedLpJEPA` in v0. Subclass `BaseMethod` directly.

Reason:

- the new method does not use RDMReg,
- it does not use rectified targets,
- it should have the smallest possible dependency surface.

## 10.2 Learnable modules

The method should create exactly these trainable modules:

```python
self.projector
self.align_head
self.projector_classifier   # only if enabled
```

The teacher is **not** trainable.

## 10.3 Learnable params property

Return:

```python
super().learnable_params + [
    {"name": "projector", "params": self.projector.parameters()},
    {"name": "align_head", "params": self.align_head.parameters()},
]
```

Add projector classifier params if needed using the repo’s existing pattern.

---

## 11. Forward function pseudocode

```python
def forward(self, X):
    out = super().forward(X)
    z = self.projector(out["feats"])

    z_c = z[:, : self.compatible_dim]
    z_f = z[:, self.compatible_dim :]

    out.update({
        "z": z,
        "z_c": z_c,
        "z_f": z_f,
    })

    if self.projector_classifier is not None:
        out["projector_logits"] = self.projector_classifier(z.detach())

    return out
```

---

## 12. Training step pseudocode

```python
def training_step(self, batch, batch_idx):
    out = super().training_step(batch, batch_idx)
    class_loss = out["loss"]

    z1, z2 = out["z"]
    z_c1, z_c2 = out["z_c"]
    z_f1, z_f2 = out["z_f"]

    _, X, targets = batch
    X = [X] if isinstance(X, torch.Tensor) else X
    x1, x2 = X[:2]

    a1 = F.normalize(self.align_head(z_c1), dim=1)
    a2 = F.normalize(self.align_head(z_c2), dim=1)

    with torch.no_grad():
        t1 = F.normalize(self.teacher(x1), dim=1)
        t2 = F.normalize(self.teacher(x2), dim=1)

    ssl_loss, pred_loss, teacher_loss, free_loss = split_teacher_sigjepa_loss(
        z1=z1,
        z2=z2,
        a1=a1,
        a2=a2,
        t1=t1,
        t2=t2,
        z_f1=z_f1,
        z_f2=z_f2,
        global_step=self.global_step,
        lambda_pred=self.lambda_pred,
        lambda_teacher=self.lambda_teacher,
        lambda_sigreg=self.lambda_sigreg,
        num_slices=self.sigreg_num_slices,
        num_points=self.sigreg_num_points,
        t_min=self.sigreg_t_min,
        t_max=self.sigreg_t_max,
    )

    projector_class_loss = torch.tensor(0.0, device=self.device)
    if self.projector_classifier is not None:
        proj_metrics1 = self._projector_classifier_step(z1, targets)
        proj_metrics2 = self._projector_classifier_step(z2, targets)
        if proj_metrics1 and proj_metrics2:
            projector_class_loss = 0.5 * (proj_metrics1["proj_loss"] + proj_metrics2["proj_loss"])
            self.log("train_proj_loss", projector_class_loss, on_epoch=True, sync_dist=True)
            self.log("train_proj_acc1", 0.5 * (proj_metrics1["proj_acc1"] + proj_metrics2["proj_acc1"]), on_epoch=True, sync_dist=True)

    self.log("train_split_teacher_sigjepa_loss", ssl_loss, on_epoch=True, sync_dist=True)
    self.log("train_pred_loss", pred_loss, on_epoch=True, sync_dist=True)
    self.log("train_teacher_loss", teacher_loss, on_epoch=True, sync_dist=True)
    self.log("train_sigreg_loss", free_loss, on_epoch=True, sync_dist=True)
    self.log("train_teacher_cosine", 0.5 * ((a1 * t1).sum(dim=1).mean() + (a2 * t2).sum(dim=1).mean()), on_epoch=True, sync_dist=True)

    return ssl_loss + class_loss + projector_class_loss
```

---

## 13. Config skeleton

```yaml
name: "split-teacher-sigjepa-imagenet100"
method: "split_teacher_sigjepa"

backbone:
  name: "resnet50"

method_kwargs:
  proj_hidden_dim: 2048
  proj_output_dim: 2048
  compatible_dim: 512
  free_dim: 1536

  lambda_pred: 1.0
  lambda_teacher: 1.0
  lambda_sigreg: 0.05

  sigreg_num_slices: 256
  sigreg_num_points: 17
  sigreg_t_min: -5.0
  sigreg_t_max: 5.0
  sigreg_use_real: false

  teacher_source: "checkpoint"
  teacher_ckpt_path: "/path/to/teacher.ckpt"
  teacher_name: null
  teacher_feature_key: null
  teacher_output_dim: null

  projector_type: "mlp"
  add_projector_classifier: true
  logging_interval: 50

data:
  dataset: imagenet100
  train_path: "/imagenet100_real/train"
  val_path: "/imagenet100_real/val"
  preload: true
  format: "image_folder"
  num_workers: 8

augmentations:
  - rrc:
      enabled: true
      crop_min_scale: 0.2
      crop_max_scale: 1.0
    color_jitter:
      enabled: true
      brightness: 0.4
      contrast: 0.4
      saturation: 0.2
      hue: 0.1
      prob: 0.8
    grayscale:
      enabled: true
      prob: 0.2
    gaussian_blur:
      enabled: true
      prob: 0.5
    solarization:
      enabled: true
      prob: 0.1
    equalization:
      enabled: false
      prob: 0.0
    horizontal_flip:
      enabled: true
      prob: 0.5
    crop_size: 224
    num_crops: 2

optimizer:
  name: "adamw"
  batch_size: 128
  lr: 0.0825
  classifier_lr: 0.0275
  weight_decay: 1e-4
  kwargs:
    clip_lr: true
    eta: 0.02
    exclude_bias_n_norm: true

scheduler:
  name: "warmup_cosine"

max_epochs: 1000
devices: [0]
sync_batchnorm: true
accelerator: "gpu"
strategy: "ddp"
precision: 16-mixed
```

---

## 14. Benchmarks and baselines

The benchmark suite should answer two questions:

1. does splitting help over full-latent distillation?
2. does SIGReg on `z_f` help over teacher-only JEPA distillation?

## 14.1 Primary metrics

For every run, report:

- ImageNet-100 linear probe top-1 on **encoder features**,
- ImageNet-100 linear probe top-1 on **projector features**,
- teacher-student cosine on `z_c`,
- average `SIGReg(z_f)`,
- train wall-clock time,
- peak GPU memory,
- optional kNN accuracy if already wired in the repo.

## 14.2 Secondary transfer metrics

Reuse the repo transfer protocol where possible:

- DTD
- CIFAR-10
- CIFAR-100
- Flowers102
- Food101
- Pets

Run frozen-feature linear probing on both encoder and projector features if compute allows.

## 14.3 Baselines to run in the same repo

### B0. Existing repo baseline

- `rectified_lpjepa`
- unchanged

This is the current codebase baseline.

### B1. Plain SIGReg JEPA (no teacher, no split)

Use the same projector and predictive loss, but regularize the **full** latent `z` with SIGReg:

```text
L = pred(z) + sigreg(z)
```

Method name suggestion:

`sigjepa`

This isolates the effect of splitting and teacher alignment.

### B2. Full-latent teacher baseline

No split. Align the **full** latent to the teacher:

```text
L = pred(z) + teacher(z)
```

Method name suggestion:

`teacher_jepa_full`

This is the most important distillation baseline.

### B3. Full-latent teacher + SIGReg baseline

No split. Teacher + SIGReg on the full latent:

```text
L = pred(z) + teacher(z) + sigreg(z)
```

Method name suggestion:

`teacher_sigjepa_full`

### B4. Split, no teacher

Keep the split, but set `lambda_teacher = 0`:

```text
L = pred(cat(z_c, z_f)) + sigreg(z_f)
```

This tests whether the free branch alone already helps.

### B5. Split, no SIGReg

Keep the split, but set `lambda_sigreg = 0`:

```text
L = pred(cat(z_c, z_f)) + teacher(z_c)
```

This tests whether the free branch needs explicit regularization.

### B6. Proposed method

```text
L = pred(cat(z_c, z_f)) + teacher(z_c) + sigreg(z_f)
```

This is the main method.

## 14.4 Optional teacher-source comparison

If multiple teacher checkpoints are available, rerun B2/B3/B6 with:

- large JEPA teacher,
- DINO-style teacher,
- other frozen foundation model teacher.

Keep the student, budget, and hyperparameters unchanged.

---

## 15. Minimal ablation grid

Keep the grid small.

## 15.1 Split size

```text
compatible_dim/free_dim:
256 / 1792
512 / 1536   <-- default
1024 / 1024
```

## 15.2 Teacher weight

```text
lambda_teacher in {0.5, 1.0, 2.0}
```

## 15.3 SIGReg weight

```text
lambda_sigreg in {0.01, 0.05, 0.1}
```

## 15.4 SIGReg slices

```text
num_slices in {64, 256, 1024}
```

Do not sweep more than this in the first pass.

---

## 16. Logging

Log these every epoch:

```text
train_split_teacher_sigjepa_loss
train_pred_loss
train_teacher_loss
train_sigreg_loss
train_teacher_cosine
train_zc_norm
train_zf_norm
train_proj_loss
train_proj_acc1
```

Recommended extra diagnostics:

```text
train_var_zc
train_var_zf
train_cov_full_z
train_sigreg_zf1
train_sigreg_zf2
```

Do not add too many custom diagnostics in v0.

---

## 17. Sanity checks

## 17.1 Reduction checks

### Check A: no compatible branch

Set:

```text
compatible_dim = 0
free_dim = 2048
lambda_teacher = 0
```

This should reduce to plain full-latent SIGReg JEPA.

### Check B: no free branch

Set:

```text
compatible_dim = 2048
free_dim = 0
lambda_sigreg = 0
```

This should reduce to full-latent teacher JEPA.

### Check C: split but no teacher

Set:

```text
lambda_teacher = 0
```

The method should still train normally.

### Check D: split but no SIGReg

Set:

```text
lambda_sigreg = 0
```

The method should still train normally.

## 17.2 Shape assertions

Add explicit assertions in the method init:

```python
assert compatible_dim > 0
assert free_dim > 0
assert compatible_dim + free_dim == proj_output_dim
```

Also assert teacher dim once inferred.

---

## 18. Recommended execution order

### Phase 1: implementation smoke test

Run for `10` to `20` epochs on ImageNet-100 with:

- `compatible_dim = 512`
- `free_dim = 1536`
- `lambda_pred = 1.0`
- `lambda_teacher = 1.0`
- `lambda_sigreg = 0.05`
- `num_slices = 64`

Goal: verify shapes, DDP, AMP, teacher loading, and logs.

### Phase 2: short benchmark

Run `100` epochs for:

- B1 plain SIGReg JEPA
- B2 full teacher
- B5 split no SIGReg
- B6 proposed

Goal: confirm the split design is worth pursuing.

### Phase 3: full benchmark

Run `1000` epochs for:

- B0 to B6
- split-size sweep
- small teacher-weight and SIGReg-weight sweeps

---

## 19. Bottom line

The **only** thing v0 should test is:

> Can a student latent be split into a teacher-compatible subspace `z_c` and a free subspace `z_f`, with teacher loss on `z_c`, SIGReg on `z_f`, and JEPA predictive loss on the full concatenated latent?

If this minimal version works, then later versions can add:

- relation matching,
- decoupling losses,
- schedules,
- better teacher targets,
- more structured split heads.

But none of those belong in the first implementation.
