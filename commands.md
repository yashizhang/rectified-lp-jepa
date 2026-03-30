```
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=rectified_lpjepa_imagenet.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true # set to false for debugging


CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true 

# To precompute 1 epoch of teacher embeddings
CUDA_VISIBLE_DEVICES=3 python3 main_prefetch_teacher.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.cache_dir=teacher_prefetch_cache/sigjepa_prefetch_1ep \
  ++method_kwargs.teacher_prefetch.num_epochs=1 \
  ++method_kwargs.teacher_prefetch.dtype=float16
```