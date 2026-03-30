```
# Baseline
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=rectified_lpjepa_imagenet.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true # set to false for debugging

# No pre-compute sigjepa
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true 

# To precompute 1 epoch of teacher embeddings
CUDA_VISIBLE_DEVICES=0 python3 main_prefetch_teacher.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.cache_dir=teacher_prefetch_cache/sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.num_epochs=1 \
  ++method_kwargs.teacher_prefetch.dtype=float16

# Train for 1 epoch using prefetched embeddings
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name split_teacher_sigjepa_imagenet100.yaml \
  name=sigjepa_train_1ep_prefetch \
  max_epochs=1 \
  ++method_kwargs.teacher_prefetch.enabled=true \
  ++method_kwargs.teacher_prefetch.cache_dir=teacher_prefetch_cache/sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.num_epochs=1 \
  ++method_kwargs.teacher_prefetch.epoch_mode=strict \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true

# To precompute 100 epoch of teacher embeddings 
CUDA_VISIBLE_DEVICES=0 python3 main_prefetch_teacher.py \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name split_teacher_sigjepa_imagenet100.yaml \
  name=sigjepa_prefetch_100ep \
  max_epochs=100 \
  ++method_kwargs.teacher_prefetch.cache_dir=teacher_prefetch_cache/sigjepa_prefetch_100ep \
  ++method_kwargs.teacher_prefetch.num_epochs=100 \
  ++method_kwargs.teacher_prefetch.dtype=float16 \
  ++method_kwargs.teacher_prefetch.overwrite=false

# Train for 100 epochs using that prefetched cache
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100 \
  --config-name split_teacher_sigjepa_imagenet100.yaml \
  name=sigjepa_train_100ep_prefetch \
  max_epochs=100 \
  ++method_kwargs.teacher_prefetch.enabled=true \
  ++method_kwargs.teacher_prefetch.cache_dir=teacher_prefetch_cache/sigjepa_prefetch_100ep \
  ++method_kwargs.teacher_prefetch.num_epochs=100 \
  ++method_kwargs.teacher_prefetch.epoch_mode=strict \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true
```