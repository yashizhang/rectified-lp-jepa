```bash
# Baseline Rectified LpJEPA
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=rectified_lpjepa_imagenet.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# Split-teacher SIGJEPA with the default ViT-B/14 student and DinoV2 ViT-L/14 teacher.
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# Optional: swap in a DinoV2-with-registers checkpoint while keeping the same student recipe.
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  ++method_kwargs.teacher_backend=hf_dinov2_with_registers \
  ++method_kwargs.teacher_model_id=facebook/dinov2-with-registers-large \
  ++method_kwargs.teacher_pooling=cls \
  ++method_kwargs.teacher_output_dim=1024 \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# Prefetch 1 epoch of DinoV2 teacher embeddings into a storage-backed cache directory.
CUDA_VISIBLE_DEVICES=0 python3 main_prefetch_teacher.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=dinov2_sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.cache_dir=./teacher_prefetch_cache/dinov2_sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.num_epochs=1 \
  ++method_kwargs.teacher_prefetch.dtype=float16

# Train for 1 epoch using the prefetched DinoV2 cache.
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name split_teacher_sigjepa_imagenet100.yaml \
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
  --config-path scripts/pretrain/imagenet-100 \
  --config-name split_teacher_sigjepa_imagenet100.yaml \
  name=dinov2_sigjepa_prefetch_100ep \
  max_epochs=100 \
  ++method_kwargs.teacher_prefetch.cache_dir=./teacher_prefetch_cache/dinov2_sigjepa_prefetch_100ep \
  ++method_kwargs.teacher_prefetch.num_epochs=100 \
  ++method_kwargs.teacher_prefetch.dtype=float16 \
  ++method_kwargs.teacher_prefetch.overwrite=false

# Train for 100 epochs using the prefetched DinoV2 cache.
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name split_teacher_sigjepa_imagenet100.yaml \
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
  --config-path scripts/pretrain/imagenet-100 \
  --config-name split_teacher_sigjepa_imagenet100.yaml \
  name=dinov2_sigjepa_prefetch_1000ep \
  max_epochs=1000 \
  ++method_kwargs.teacher_prefetch.cache_dir=./teacher_prefetch_cache/dinov2_sigjepa_prefetch_1000ep \
  ++method_kwargs.teacher_prefetch.num_epochs=1000 \
  ++method_kwargs.teacher_prefetch.dtype=float16 \
  ++method_kwargs.teacher_prefetch.overwrite=false

torchrun --nproc_per_node=4 main_prefetch_teacher.py \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name split_teacher_sigjepa_imagenet100.yaml \
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
  --config-path scripts/pretrain/imagenet-100 \
  --config-name split_teacher_sigjepa_imagenet100.yaml \
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
