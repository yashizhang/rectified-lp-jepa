# Split-Teacher SIGJEPA pseudocode

This pseudocode matches the current `split_teacher_sigjepa` training implementation in this repo.

## Model setup

```text
Input:
    backbone f_θ
    projector g_θ : features_dim -> proj_output_dim
    align head h_θ : compatible_dim -> teacher_dim
    frozen teacher T (online) OR prefetched teacher cache C

Hyperparameters:
    compatible_dim, free_dim
    lambda_pred, lambda_teacher, lambda_sigreg
    SIGReg settings: num_slices, num_points, t_min, t_max, sigreg_use_real
```

## One training epoch

```text
for epoch in {0, ..., max_epochs - 1}:
    if teacher prefetch cache is enabled:
        cache_epoch <- resolve_teacher_prefetch_epoch(
            train_epoch=epoch,
            num_prefetch_epochs=C.num_epochs,
            epoch_mode in {wrap, strict}
        )
        set deterministic dataset epoch seed to cache_epoch
        open prefetched teacher file for cache_epoch

    for batch = (img_indexes, [x1, x2], targets) from train_loader:
        # ---------------------------------------------
        # 1) Student forward on both large crops
        # ---------------------------------------------
        feats1 <- f_θ(x1)
        feats2 <- f_θ(x2)

        z1 <- g_θ(feats1)
        z2 <- g_θ(feats2)

        z_c1 <- z1[:, :compatible_dim]
        z_c2 <- z2[:, :compatible_dim]
        z_f1 <- z1[:, compatible_dim:]
        z_f2 <- z2[:, compatible_dim:]

        # ---------------------------------------------
        # 2) Online linear probe / projector classifier
        #    (inherited from BaseMethod)
        # ---------------------------------------------
        class_loss <- classifier_loss(feats1, feats2, targets)
        projector_class_loss <- optional_projector_classifier_loss(z1, z2, targets)

        # ---------------------------------------------
        # 3) Teacher-compatible branch on z_c only
        # ---------------------------------------------
        if lambda_teacher > 0 and compatible_dim > 0:
            a1 <- normalize(h_θ(z_c1))
            a2 <- normalize(h_θ(z_c2))

            if teacher prefetch cache is enabled:
                t1, t2 <- C[cache_epoch, img_indexes, view=0/1]
                t1 <- normalize(t1)
                t2 <- normalize(t2)
            else:
                t1 <- normalize(T(x1))
                t2 <- normalize(T(x2))
        else:
            a1, a2, t1, t2 <- None

        # ---------------------------------------------
        # 4) Free branch on z_f only
        # ---------------------------------------------
        if lambda_sigreg > 0 and free_dim > 0:
            free_loss <- SIGReg(z_f1) + SIGReg(z_f2)
        else:
            free_loss <- 0

        # ---------------------------------------------
        # 5) JEPA-style predictive loss on the full latent
        # ---------------------------------------------
        pred_loss <- predictive_loss(z1, z2)

        # ---------------------------------------------
        # 6) Total SSL loss
        # ---------------------------------------------
        teacher_loss <- alignment_loss(a1, a2, t1, t2)

        ssl_loss <- (
            lambda_pred    * pred_loss
          + lambda_teacher * teacher_loss
          + lambda_sigreg  * free_loss
        )

        total_loss <- class_loss + projector_class_loss + ssl_loss

        # ---------------------------------------------
        # 7) Optimize student only
        # ---------------------------------------------
        backprop(total_loss)
        optimizer.step()
        optimizer.zero_grad()

        # Frozen teacher / prefetched cache are never updated.
```

## Offline teacher-prefetch pass

```text
Build the same deterministic augmentation pipeline used by training.

for cache_epoch in {0, ..., num_prefetch_epochs - 1}:
    set deterministic dataset epoch seed to cache_epoch
    for batch = (img_indexes, [x1, x2], _) from train_loader_without_shuffle:
        t1 <- normalize(T(x1))
        t2 <- normalize(T(x2))
        save [t1, t2] at cache[cache_epoch, img_indexes]
```

## Key contracts

```text
- predictive loss uses the full latent cat([z_c, z_f]) implicitly via z
- teacher loss touches z_c only
- SIGReg touches z_f only
- teacher is either:
    (a) online frozen I-JEPA, or
    (b) an on-disk prefetched cache generated from the exact same deterministic crops
- cached teacher targets are indexed by (cache_epoch, sample_index, view_index)
```
