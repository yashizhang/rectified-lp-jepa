```
CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=rectified_lpjepa_imagenet.yaml \
  name=lpjepa_baseline \ # wandb run name 
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true # set to false for debugging


CUDA_VISIBLE_DEVICES=0,1,2 python3 main_pretrain.py \
  --config-path scripts/pretrain/imagenet-100/ \
  --config-name=split_teacher_sigjepa_imagenet100.yaml \
  name=sigjepa \ # wandb run name 
  ++wandb.entity=yashi-zhang \
  ++wandb.project=jepa \
  ++wandb.enabled=true 
```