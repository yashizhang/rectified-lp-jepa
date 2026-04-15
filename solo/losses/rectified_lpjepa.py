# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import math
import torch
import torch.nn.functional as F
from solo.utils.misc import gather
import mpmath as mp
from torch.distributions.laplace import Laplace

from solo.utils.metrics import (
    l1_sparsity_metric,
    l0_sparsity_metric,
    variance_loss,
    covariance_loss,
)

# =========================
# Invariance / Prediction Losses
# =========================
def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    """
    return F.mse_loss(z1, z2)


def multicrop_predictive_loss(global_views, all_views) -> torch.Tensor:
    """LeJEPA-style predictive loss.

    The global views are averaged into a per-sample center and every view
    (global + local) predicts that center.
    """

    if len(global_views) == 2 and len(all_views) == 2:
        # Preserve the original 2-view behavior for backward compatibility.
        return invariance_loss(global_views[0], global_views[1])

    if len(global_views) == 0:
        raise ValueError("At least one global view is required.")
    if len(all_views) == 0:
        raise ValueError("At least one view is required.")

    centers = torch.stack(global_views, dim=0).mean(dim=0)
    stacked_views = torch.stack(all_views, dim=0)
    return F.mse_loss(stacked_views, centers.unsqueeze(0).expand_as(stacked_views))

# =========================
# Generalized Gaussian Distribution Sampling
# =========================
def sample_product_laplace(shape, device, dtype, loc=0.0, scale=1/math.sqrt(2)):
    """
    Sample from the product Laplace distribution directly on the target device
    using torch.distributions.Laplace.
    """
    # shape is (B, D)
    loc_t = torch.tensor(loc, device=device, dtype=dtype)
    scale_t = torch.tensor(scale, device=device, dtype=dtype)
    laplace_dist = Laplace(loc=loc_t, scale=scale_t)
    return laplace_dist.sample(shape)

def sample_lp_distribution(shape, p, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """
    Samples from the Generalized Gaussian distribution GN_p(loc, scale).
    Optimized for p=1.0 (Laplace) and p=2.0 (Gaussian).
    """
    if p == 1.0:
        return sample_product_laplace(shape, device, dtype, loc=loc, scale=scale)
    elif p == 2.0:
        # GN_2.0 is Gaussian. Note: GN_p(loc, scale) has variance scale^2 for p=2.0
        return loc + scale * torch.randn(shape, device=device, dtype=dtype)
    else:
        # Generic slow sampling logic
        sign = torch.empty(shape, device=device, dtype=dtype).bernoulli_(0.5)
        sign = 2 * sign - 1
        gamma = torch.distributions.Gamma(concentration=1.0/p, rate=1.0)
        g = gamma.sample(shape).to(device=device, dtype=dtype)
        x = sign * (p * g).pow(1.0 / p)
        return loc + scale * x

def determine_sigma_for_lp_dist(p):
    """
    Determines the scale parameter sigma such that the Generalized Gaussian GN_p(0, sigma)
    has unit variance.
    """
    sigma = (math.gamma(1/p)**(1/2)) / ((p ** (1/p)) * (math.gamma(3/p)**(1/2)))
    return sigma

# =========================
# Rectified Generalized Gaussian Moments
# =========================
def rectified_gengaus_mean_var_unified(p, mu, sigma):
    """
    Unified (non-piecewise) mean/var for Y = ReLU(X),
    X ~ GN_p(mu, sigma) with pdf ∝ exp(-|x-mu|^p / (p*sigma^p)).
    Uses sign(mu) + lower/upper incomplete gamma functions.
    Uses mpmath for high-precision incomplete gamma functions.
    """
    p = mp.mpf(p)
    mu = mp.mpf(mu)
    sigma = mp.mpf(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if p <= 0:
        raise ValueError("p must be > 0")

    sgn = mp.sign(mu)  # -1, 0, +1
    s1 = mp.mpf(1) / p
    s2 = mp.mpf(2) / p
    s3 = mp.mpf(3) / p

    t = (abs(mu) ** p) / (p * (sigma ** p))
    G1 = mp.gamma(s1)

    # lower incomplete gammas
    lower1 = mp.gammainc(s1, 0, t)         # γ(1/p, t)
    lower3 = mp.gammainc(s3, 0, t)         # γ(3/p, t)

    # upper incomplete gamma
    upper2 = mp.gammainc(s2, t, mp.inf)    # Γ(2/p, t)

    # unified coefficients
    A = (G1 + sgn * lower1) / G1
    B = upper2 / G1
    C = (mp.gamma(s3) + sgn * lower3) / G1

    p1 = p ** (mp.mpf(1) / p)
    p2 = p ** (mp.mpf(2) / p)

    EY  = mp.mpf('0.5') * (mu * A + p1 * sigma * B)
    EY2 = mp.mpf('0.5') * (mu**2 * A + 2*mu*p1*sigma*B + p2 * sigma**2 * C)
    VarY = EY2 - EY**2
    return float(EY), float(VarY)

def choose_sigma_for_unit_var(p, mu, target_var=1.0, rtol=1e-10, max_iter=2000):
    """
    Solve for sigma>0 such that Var(ReLU(X)) = target_var where X~GN_p(mu,sigma).
    Robust bisection on f(sigma)=Var- target_var.
    """
    p = mp.mpf(p); mu = mp.mpf(mu); target_var = mp.mpf(target_var)

    def f(sig):
        return rectified_gengaus_mean_var_unified(p, mu, sig)[1] - target_var

    # --- bracket a root ---
    lo = mp.mpf('1e-8')
    hi = mp.mpf('1.0')
    flo = f(lo)
    fhi = f(hi)

    # Increase hi until sign change
    k = 0
    while flo * fhi > 0 and k < 200:
        hi *= 2
        fhi = f(hi)
        k += 1

    if flo * fhi > 0:
        raise RuntimeError("Failed to bracket a root for sigma. Try different initial range.")

    # --- bisection ---
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fmid = f(mid)

        if abs(fmid) <= rtol * (1 + abs(target_var)):
            return float(mid)

        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return float((lo + hi) / 2)

# =========================
# Sliced Wasserstein Distance (RDMReg)
# =========================
def sliced_wasserstein_distance_for_one_view(features, projection_vectors, target_dist, mean_shift_value=0.0, lp_norm_parameter=1.0, chosen_sigma=None):
    """
    Computes the 1D Wasserstein distance between projected features and projected target samples.
    """
    B, D = features.shape
    projected_features = torch.matmul(features, projection_vectors.T)

    # Sample from the target distribution (Rectified or not)
    if target_dist == "rectified_lp_distribution":
        target_samples = torch.relu(sample_lp_distribution(
            shape=(B, D),
            p=lp_norm_parameter,
            loc=mean_shift_value,
            scale=chosen_sigma,
            device=features.device,
            dtype=features.dtype
        ))
    elif target_dist == "lp_distribution":
        target_samples = sample_lp_distribution(
            shape=(B, D),
            p=lp_norm_parameter,
            loc=mean_shift_value,
            scale=chosen_sigma,
            device=features.device,
            dtype=features.dtype
        )
    else:
        raise ValueError(f"Unsupported target_dist: {target_dist}")

    projected_targets = torch.matmul(target_samples, projection_vectors.T)
    
    # Sort along batch dimension for each projection
    sorted_features, _ = torch.sort(projected_features, dim=0)
    sorted_targets, _ = torch.sort(projected_targets, dim=0)
    
    # Compute L2 distance between sorted projections
    wasserstein_1d = torch.mean((sorted_features - sorted_targets)**2, dim=0)
    return torch.mean(wasserstein_1d)

def rdmreg_loss(z1, z2, projection_vectors, target_dist, mean_shift_value=0.0, lp_norm_parameter=1.0, chosen_sigma=None):
    """Backward-compatible 2-view RDMReg wrapper."""
    return rdmreg_loss_multiview(
        [z1, z2],
        projection_vectors,
        target_dist,
        mean_shift_value=mean_shift_value,
        lp_norm_parameter=lp_norm_parameter,
        chosen_sigma=chosen_sigma,
    )


def rdmreg_loss_multiview(views, projection_vectors, target_dist, mean_shift_value=0.0, lp_norm_parameter=1.0, chosen_sigma=None):
    """Computes the Sliced Wasserstein Distance (RDMReg) loss across an arbitrary number of views."""

    if len(views) == 0:
        raise ValueError("At least one view is required for RDMReg.")

    if isinstance(projection_vectors, list):
        if len(projection_vectors) != len(views):
            raise ValueError(
                f"Expected one projection matrix per view, got {len(projection_vectors)} for {len(views)} views."
            )
        losses = [
            sliced_wasserstein_distance_for_one_view(view, proj, target_dist, mean_shift_value, lp_norm_parameter, chosen_sigma)
            for view, proj in zip(views, projection_vectors)
        ]
    elif isinstance(projection_vectors, torch.Tensor):
        losses = [
            sliced_wasserstein_distance_for_one_view(view, projection_vectors, target_dist, mean_shift_value, lp_norm_parameter, chosen_sigma)
            for view in views
        ]
    else:
        raise ValueError("Invalid projection_vectors type")

    return torch.stack(losses).mean()

# =========================
# Main Rectified LpJEPA Loss
# =========================
def rectified_lp_jepa_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    projection_vectors: torch.Tensor,
    target_distribution: str,
    invariance_loss_weight: float,
    rdm_reg_loss_weight: float,
    mean_shift_value: float = 0.0,
    lp_norm_parameter: float = 1.0,
    chosen_sigma: float = None,
):
    """Backward-compatible 2-view Rectified LpJEPA loss."""
    return rectified_lp_jepa_multicrop_loss(
        global_views=[z1, z2],
        all_views=[z1, z2],
        gathered_views=None,
        projection_vectors=projection_vectors,
        target_distribution=target_distribution,
        invariance_loss_weight=invariance_loss_weight,
        rdm_reg_loss_weight=rdm_reg_loss_weight,
        mean_shift_value=mean_shift_value,
        lp_norm_parameter=lp_norm_parameter,
        chosen_sigma=chosen_sigma,
    )


# =========================
# Multi-crop Rectified LpJEPA Loss
# =========================
def rectified_lp_jepa_multicrop_loss(
    global_views,
    all_views,
    projection_vectors,
    target_distribution: str,
    invariance_loss_weight: float,
    rdm_reg_loss_weight: float,
    mean_shift_value: float = 0.0,
    lp_norm_parameter: float = 1.0,
    chosen_sigma: float = None,
    gathered_views=None,
):
    """Computes the LeJEPA-style multi-crop Rectified LpJEPA loss.

    Args:
        global_views: sequence of global-view embeddings.
        all_views: sequence of all view embeddings (global + local).
        gathered_views: optional gathered version of ``all_views`` for RDMReg.
            If omitted, each view is gathered internally, matching the original behavior.
    """

    sim_loss = multicrop_predictive_loss(global_views, all_views)

    if gathered_views is None:
        reg_views = [gather(view) for view in all_views]
    else:
        reg_views = list(gathered_views)

    reg_loss = rdmreg_loss_multiview(
        reg_views,
        projection_vectors,
        target_distribution,
        mean_shift_value=mean_shift_value,
        lp_norm_parameter=lp_norm_parameter,
        chosen_sigma=chosen_sigma,
    )

    loss = (invariance_loss_weight * sim_loss) + (rdm_reg_loss_weight * reg_loss)
    return loss, sim_loss, reg_loss
