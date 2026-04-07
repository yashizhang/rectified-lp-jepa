```bash
# B0. Existing repo baseline: Rectified LpJEPA
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=rectified_lpjepa_imagenet.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# B1. Plain SIGReg JEPA (config-only via split_teacher_sigjepa with compatible_dim=0, free_dim=2048)
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=sigjepa_imagenet100.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# B2. Full-latent teacher baseline (config-only via split_teacher_sigjepa with compatible_dim=2048, free_dim=0)
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=teacher_jepa_full_imagenet100.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# B3. Full-latent teacher + SIGReg baseline (thin dedicated method)
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=teacher_sigjepa_full_imagenet100.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# B4. Split, no teacher
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_no_teacher_imagenet100.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# B5. Split, no SIGReg
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_no_sigreg_imagenet100.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# B6. Proposed split-teacher SIGJEPA with the default ViT-B/14 student and DinoV2 ViT-L/14 teacher.
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# Optional: explicit B6 alias config with the same defaults.
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_b6_imagenet100.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# Optional: swap in a DinoV2-with-registers checkpoint while keeping the same student recipe.
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  ++method_kwargs.teacher_backend=hf_dinov2_with_registers \
  ++method_kwargs.teacher_model_id=facebook/dinov2-with-registers-large \
  ++method_kwargs.teacher_pooling=cls \
  ++method_kwargs.teacher_output_dim=1024 \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# Split-size sweep configs available:
#   split_teacher_sigjepa_split_c256_f1792_imagenet100.yaml
#   split_teacher_sigjepa_split_c1792_f256_imagenet100.yaml
# Teacher-weight sweep configs available:
#   split_teacher_sigjepa_lambda_teacher_0p5_imagenet100.yaml
#   split_teacher_sigjepa_lambda_teacher_2p0_imagenet100.yaml
# SIGReg-weight sweep configs available:
#   split_teacher_sigjepa_lambda_sigreg_0p01_imagenet100.yaml
#   split_teacher_sigjepa_lambda_sigreg_0p1_imagenet100.yaml
# SIGReg-slice sweep configs available:
#   split_teacher_sigjepa_num_slices_64_imagenet100.yaml
#   split_teacher_sigjepa_num_slices_1024_imagenet100.yaml

# Prefetch 1 epoch of DinoV2 teacher embeddings into a storage-backed cache directory.
CUDA_VISIBLE_DEVICES=0 python3 main_prefetch_teacher.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=dinov2_sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.cache_dir=./teacher_prefetch_cache/dinov2_sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.num_epochs=1 \
  ++method_kwargs.teacher_prefetch.dtype=float16

# Train for 1 epoch using the prefetched DinoV2 cache.
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=dinov2_sigjepa_train_1ep_prefetch \
  max_epochs=1 \
  ++method_kwargs.teacher_prefetch.enabled=true \
  ++method_kwargs.teacher_prefetch.cache_dir=./teacher_prefetch_cache/dinov2_sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.num_epochs=1 \
  ++method_kwargs.teacher_prefetch.epoch_mode=strict \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# Prefetch 100 epochs of DinoV2 teacher embeddings to shared/local storage.
CUDA_VISIBLE_DEVICES=0 python3 main_prefetch_teacher.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=dinov2_sigjepa_prefetch_100ep \
  max_epochs=100 \
  ++method_kwargs.teacher_prefetch.cache_dir=./teacher_prefetch_cache/dinov2_sigjepa_prefetch_100ep \
  ++method_kwargs.teacher_prefetch.num_epochs=100 \
  ++method_kwargs.teacher_prefetch.dtype=float16 \
  ++method_kwargs.teacher_prefetch.overwrite=false

# Train for 100 epochs using the prefetched DinoV2 cache.
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=dinov2_sigjepa_train_100ep_prefetch \
  max_epochs=100 \
  ++method_kwargs.teacher_prefetch.enabled=true \
  ++method_kwargs.teacher_prefetch.cache_dir=./teacher_prefetch_cache/dinov2_sigjepa_prefetch_100ep \
  ++method_kwargs.teacher_prefetch.num_epochs=100 \
  ++method_kwargs.teacher_prefetch.epoch_mode=strict \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# Prefetch 1000 epochs of DinoV2 teacher embeddings to shared/local storage.
torchrun --nproc_per_node=4 main_prefetch_teacher.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=dinov2_sigjepa_prefetch_1000ep \
  max_epochs=1000 \
  data.preload=false \
  ++method_kwargs.teacher_prefetch.num_workers=2 \
  ++method_kwargs.teacher_prefetch.cache_dir=/gpfs/projects/AI4D/core-132/yashi/teacher_prefetch_cache/dinov2_sigjepa_prefetch_1000ep \
  ++method_kwargs.teacher_prefetch.num_epochs=1000 \
  ++method_kwargs.teacher_prefetch.dtype=float16 \
  ++method_kwargs.teacher_prefetch.overwrite=false

# Train for 1000 epochs using the prefetched DinoV2 cache.
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path=scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=dinov2_sigjepa_train_1000ep_prefetch \
  max_epochs=1000 \
  ++method_kwargs.teacher_prefetch.enabled=true \
  ++method_kwargs.teacher_prefetch.cache_dir=./teacher_prefetch_cache/dinov2_sigjepa_prefetch_1000ep \
  ++method_kwargs.teacher_prefetch.num_epochs=1000 \
  ++method_kwargs.teacher_prefetch.epoch_mode=strict \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true
```
