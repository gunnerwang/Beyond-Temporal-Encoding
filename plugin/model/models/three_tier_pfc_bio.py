"""
Three-Tier PFC Bio Architecture: Genuine Three-Tier Hierarchy

Tier 1 — Slow Prior (LimiX, frozen):
    Base predictions + raw 192-dim embeddings from LimiX foundation model.

Tier 2 — Intermediate Regime Modulation (k-means):
    K-means clustering on raw LimiX embeddings to detect distribution regimes.
    Per-regime bias and scale learned from training error statistics.
    Note: regime bias can be disabled (use_regime_bias=False) since training
    biases often overcorrect on test due to train/test error mismatch.

Tier 3 — Fast Online Adaptation (k-NN):
    Online euclidean k-NN error correction with test-time ground truth feedback.
    Circular buffer storing (embedding, error) pairs for real-time adaptation.
    Optional whitening of embeddings (PCA or shrinkage Mahalanobis) for better distance metric.

Tier 3.5 — CfC Temporal Correction (optional):
    Closed-form Continuous-time (CfC) cell captures temporal error dynamics
    (drift, trends, regime transitions) that spatial k-NN cannot model.
    Purely additive — does not modify any existing tier.

Forward pass:
    regime_id = assign_regime(raw_embeddings)
    knn_emb = whiten(raw_embeddings) if use_whitening else raw_embeddings
    knn_correction = online_knn_lookup(knn_emb)
    effective_scale = online_scale * online_scale_by_regime[regime_id]  (if use_regime_online_scale)
    if use_regime_bias:
        output = slow_pred + regime_scale[regime_id] * effective_scale * knn_correction + regime_bias[regime_id]
    else:
        output = slow_pred + effective_scale * knn_correction
"""

import os
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List

from model.lib.slow_priors import create_slow_prior


# =============================================================================
# Tier 3.5: Minimal CfC Cell (Closed-form Continuous-time)
# =============================================================================





# =============================================================================
# Tier 2: Regime Detector (K-Means on raw LimiX embeddings)
# =============================================================================

class RegimeDetector:
    """
    K-means clustering on raw LimiX embeddings with per-regime error statistics.

    During bio-learning (single pass over training data):
    1. Run k-means (K clusters) on raw training embeddings
    2. For each cluster r, compute:
       - bias_r = mean(y_true - y_pred) for samples in cluster r
       - scale_r = std(error_r) / global_std — how variable errors are in this regime
    3. Store centroids + per-regime (scale, bias)

    At inference: assign each sample to nearest centroid, apply regime-specific modulation.
    """

    def __init__(self, n_regimes: int = 8, device: str = 'cuda'):
        self.n_regimes = n_regimes
        self.device = device

        # Fitted parameters (set by fit())
        self.centroids = None    # [n_regimes, embed_dim]
        self.biases = None       # [n_regimes]
        self.scales = None       # [n_regimes]
        self.online_scale_by_regime = None  # [n_regimes], per-regime online_scale multiplier
        # Stats for regime confidence: distance to nearest centroid
        self.centroid_dist_median = 1.0
        self.centroid_dist_mad = 1.0
        self.fitted = False

    def fit(self, embeddings: np.ndarray, errors: np.ndarray, n_iter: int = 50, subsample: Optional[int] = None):
        """
        Fit k-means on embeddings and compute per-regime error statistics.

        Args:
            embeddings: [n_samples, embed_dim] — raw LimiX embeddings
            errors: [n_samples] — prediction errors (y_true - y_pred)
            n_iter: number of k-means iterations
        """
        n_samples, embed_dim = embeddings.shape

        # Optional subsampling for faster k-means fit
        if subsample is not None and subsample > 0 and n_samples > subsample:
            rng_sub = np.random.RandomState(0)
            idx_sub = rng_sub.choice(n_samples, size=int(subsample), replace=False)
            emb_fit = embeddings[idx_sub]
            err_fit = errors[idx_sub]
        else:
            emb_fit = embeddings
            err_fit = errors

        n_fit = emb_fit.shape[0]
        k = min(self.n_regimes, n_fit)

        # K-means initialization: k-means++ style
        rng = np.random.RandomState(42)
        centroids = np.zeros((k, embed_dim))
        centroids[0] = emb_fit[rng.randint(n_fit)]

        for i in range(1, k):
            dists = np.min(
                np.sum((emb_fit[:, None, :] - centroids[None, :i, :]) ** 2, axis=2),
                axis=1
            )
            denom = dists.sum() + 1e-10
            probs = dists / denom
            # Guard against numerical issues (all-zero/NaN probabilities)
            if not np.isfinite(probs).all() or probs.sum() <= 0:
                probs = None
            centroids[i] = emb_fit[rng.choice(n_fit, p=probs)]

        # K-means iterations
        for _ in range(n_iter):
            # Assign to nearest centroid
            dists = np.sum((emb_fit[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assignments = np.argmin(dists, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for r in range(k):
                mask = assignments == r
                if mask.sum() > 0:
                    new_centroids[r] = emb_fit[mask].mean(axis=0)
                else:
                    new_centroids[r] = centroids[r]
            centroids = new_centroids

        # Final assignments
        dists = np.sum((emb_fit[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)

        # Compute per-regime error statistics (on fit subset)
        global_std = err_fit.std() + 1e-8
        biases = np.zeros(k)
        scales = np.ones(k)

        for r in range(k):
            mask = assignments == r
            if mask.sum() > 1:
                biases[r] = err_fit[mask].mean()
                scales[r] = err_fit[mask].std() / global_std
            elif mask.sum() == 1:
                biases[r] = err_fit[mask].mean()
                scales[r] = 1.0

        # Store as tensors on device
        self.centroids = torch.tensor(centroids, dtype=torch.float32, device=self.device)
        self.biases = torch.tensor(biases, dtype=torch.float32, device=self.device)
        self.scales = torch.tensor(scales, dtype=torch.float32, device=self.device)
        self.online_scale_by_regime = torch.ones(k, dtype=torch.float32, device=self.device)
        # Store centroid distance stats for confidence-style modulation
        # (distance of each point to its assigned centroid)
        try:
            dist_to_centroid = np.sqrt(dists[np.arange(n_fit), assignments])
            self.centroid_dist_median = float(np.median(dist_to_centroid))
            # robust spread (MAD) for normalization if needed
            self.centroid_dist_mad = float(np.median(np.abs(dist_to_centroid - self.centroid_dist_median)) + 1e-8)
        except Exception:
            self.centroid_dist_median = 1.0
            self.centroid_dist_mad = 1.0

        self.fitted = True

        # Print regime statistics
        print(f"RegimeDetector: fitted {k} regimes on {n_samples} samples")
        for r in range(k):
            mask = assignments == r
            n = mask.sum()
            print(f"  Regime {r}: n={n}, bias={biases[r]:.4f}, scale={scales[r]:.4f}")

    def assign(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Assign samples to nearest regime centroid.

        Args:
            embeddings: [batch, embed_dim]

        Returns:
            regime_ids: [batch] — integer regime assignments
        """
        if not self.fitted:
            return torch.zeros(embeddings.size(0), dtype=torch.long, device=embeddings.device)

        # Dimension mismatch is a bug — log warning but handle gracefully
        centroid_dim = self.centroids.size(1)
        embed_dim = embeddings.size(1)
        if embed_dim != centroid_dim:
            print(f"WARNING: regime_detector.assign dim mismatch: embeddings={embed_dim} vs centroids={centroid_dim}")
            # Truncate or zero-pad as fallback to avoid crash
            if embed_dim > centroid_dim:
                embeddings = embeddings[:, :centroid_dim]
            else:
                pad = torch.zeros(embeddings.size(0), centroid_dim - embed_dim,
                                  device=embeddings.device, dtype=embeddings.dtype)
                embeddings = torch.cat([embeddings, pad], dim=1)

        # Euclidean distance to centroids
        dists = torch.cdist(embeddings, self.centroids)  # [batch, n_regimes]
        return dists.argmin(dim=-1)

    def get_params(self, regime_ids: torch.Tensor):
        """Get per-regime scale and bias for given assignments.

        Args:
            regime_ids: [batch] — regime assignments

        Returns:
            scales: [batch]
            biases: [batch]
        """
        if not self.fitted:
            batch_size = regime_ids.size(0)
            device = regime_ids.device
            return torch.ones(batch_size, device=device), torch.zeros(batch_size, device=device)

        return self.scales[regime_ids], self.biases[regime_ids]

    def get_online_scales(self, regime_ids: torch.Tensor) -> torch.Tensor:
        """Get per-regime online_scale multipliers for given assignments.

        Args:
            regime_ids: [batch] — regime assignments

        Returns:
            online_scales: [batch]
        """
        if not self.fitted or self.online_scale_by_regime is None:
            return torch.ones(regime_ids.size(0), device=regime_ids.device)
        return self.online_scale_by_regime[regime_ids]

    def fit_online_scales(self, embeddings: np.ndarray, slow_preds: np.ndarray,
                          true_labels: np.ndarray, knn_estimates: np.ndarray,
                          global_online_scale: float, use_regime_bias: bool,
                          ridge: float = 1e-3, clip: float = 2.0):
        """Fit per-regime online_scale by closed-form ridge regression on validation data.

        For each regime r, solve for s_r minimizing:
            sum_i (y_i - base_i - s_r * est_i)^2 + ridge * s_r^2
        where base_i = slow_pred_i + regime_bias_i (if use_regime_bias),
        and est_i = global_online_scale * regime_scale_i * knn_estimate_i
        (or global_online_scale * knn_estimate_i if not use_regime_bias).

        s_r = (sum est_i * residual_i) / (sum est_i^2 + ridge)

        Args:
            embeddings: [n, embed_dim] — raw embeddings (same space as centroids)
            slow_preds: [n] — slow prior predictions
            true_labels: [n] — ground truth
            knn_estimates: [n] — kNN correction estimates (before any scaling)
            global_online_scale: the global online_scale factor
            use_regime_bias: whether regime bias is applied
            ridge: regularization strength
            clip: max absolute value for fitted scales
        """
        if not self.fitted:
            return

        n = len(embeddings)
        emb_t = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        regime_ids = self.assign(emb_t).cpu().numpy()

        scales_np = self.scales.cpu().numpy()
        biases_np = self.biases.cpu().numpy()

        fitted_scales = np.ones(self.n_regimes)

        print(f"Fitting per-regime online_scale on {n} val samples (ridge={ridge}, clip={clip}):")
        for r in range(self.n_regimes):
            mask = regime_ids == r
            n_r = mask.sum()
            if n_r == 0:
                print(f"  Regime {r}: n=0, scale=1.000 (default)")
                continue

            y_r = true_labels[mask]
            slow_r = slow_preds[mask]
            knn_r = knn_estimates[mask]

            # Compute base prediction (without kNN correction)
            if use_regime_bias:
                base_r = slow_r + biases_np[r]
                # est = global_online_scale * regime_scale_r * knn_estimate
                est_r = global_online_scale * scales_np[r] * knn_r
            else:
                base_r = slow_r
                est_r = global_online_scale * knn_r

            # Residual that kNN correction should explain
            residual_r = y_r - base_r

            # Only fit on samples where est != 0
            nonzero = np.abs(est_r) > 1e-10
            if nonzero.sum() < 2:
                print(f"  Regime {r}: n={n_r}, nonzero={nonzero.sum()}, scale=1.000 (too few)")
                continue

            est_nz = est_r[nonzero]
            res_nz = residual_r[nonzero]

            # Closed-form ridge: s = (est . res) / (est . est + ridge)
            s_r = np.dot(est_nz, res_nz) / (np.dot(est_nz, est_nz) + ridge)
            s_r = float(np.clip(s_r, 0.0, clip))
            fitted_scales[r] = s_r
            print(f"  Regime {r}: n={n_r}, nonzero={nonzero.sum()}, scale={s_r:.4f}")

        self.online_scale_by_regime = torch.tensor(
            fitted_scales, dtype=torch.float32, device=self.device
        )
        print(f"  Final scales: {fitted_scales.tolist()}")
        # store for later inspection
        self._last_fitted_online_scales = fitted_scales.tolist()


# =============================================================================
# Tier 3: Online K-NN Buffer (Euclidean distance, circular buffer)
# =============================================================================




class EnsemblePriorGating(nn.Module):
    """Sample-adaptive gating MLP for blending multiple LimiX prior variants.

    Given an embedding vector, outputs softmax weights over N prior variants.
    Each variant is a different LimiX cache (e.g., different temporal policy or seed).

    This implements Tier 2 "executive control": the PFC dynamically allocates
    attention across multiple prior sources based on the current input context.
    """

    def __init__(self, embed_dim: int, n_priors: int, hidden_dim: int = 128):
        super().__init__()
        self.n_priors = n_priors
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_priors),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, embed_dim] — whitened or raw embeddings
        Returns:
            weights: [B, n_priors] — softmax weights over prior variants
        """
        return torch.softmax(self.net(embeddings), dim=-1)






# =============================================================================

class OnlineKNNBuffer:
    """
    GPU-resident circular buffer for online k-NN error correction.

    Uses euclidean distance + inverse-distance weighting for correction lookup,
    with optional temporal decay to weight recent entries more heavily.
    Buffer stores (embedding, error) pairs from observed test-time feedback.

    Optional: store a discrete "regime_id" per entry (Tier2), and apply a
    mismatch penalty during kNN lookup.
    """

    def __init__(self, max_size: int = 2048, device: str = 'cuda', temporal_decay: float = 1.0,
                 knn_kernel: str = 'inverse', knn_agg: str = 'mean',
                 knn_trim_q: float = 0.1, knn_clip: float = 3.0,
                 knn_softmax_temp: float = 2.0,
                 regime_reweight_gamma: float = 0.0):
        self.max_size = max_size
        self.device = device
        self.temporal_decay = temporal_decay
        self.knn_kernel = knn_kernel    # 'inverse' | 'gaussian' | 'softmax'
        self.knn_softmax_temp = float(knn_softmax_temp)
        self.knn_agg = knn_agg          # 'mean' | 'trimmed' | 'clip'
        self.knn_trim_q = knn_trim_q    # fraction to trim from each tail (trimmed agg)
        self.knn_clip = knn_clip        # number of MADs for clipping (clip agg)
        self.regime_reweight_gamma = float(regime_reweight_gamma)

        # Lazy-initialized buffers
        self.keys = None      # [max_size, embed_dim]
        self.values = None    # [max_size, 1]
        self.timestamps = None  # [max_size] — write counter at time of storage
        self.regimes = None     # [max_size] — optional regime id per entry

        self.write_ptr = 0
        self.current_size = 0
        self.global_write_counter = 0

    def _init_buffers(self, embed_dim: int, device):
        """Lazy init on first write."""
        if self.keys is None:
            self.keys = torch.zeros(self.max_size, embed_dim, device=device)
            self.values = torch.zeros(self.max_size, 1, device=device)
            self.timestamps = torch.zeros(self.max_size, device=device)
            self.regimes = torch.zeros(self.max_size, dtype=torch.long, device=device)
            self.device = device

    def write(self, keys: torch.Tensor, values: torch.Tensor, regimes: Optional[torch.Tensor] = None):
        """Append (embedding, error) pairs to circular buffer.

        Args:
            keys: [batch, embed_dim]
            values: [batch, 1] or [batch]
            regimes: optional [batch] long tensor of regime ids
        """
        self._init_buffers(keys.size(-1), keys.device)
        keys = keys.detach()
        values = values.detach()
        if values.dim() == 1:
            values = values.unsqueeze(-1)

        if regimes is not None:
            regimes = regimes.detach().to(dtype=torch.long)

        batch_size = keys.size(0)
        for i in range(batch_size):
            self.keys[self.write_ptr] = keys[i]
            self.values[self.write_ptr] = values[i]
            self.timestamps[self.write_ptr] = self.global_write_counter
            if regimes is not None:
                self.regimes[self.write_ptr] = regimes[i]
            self.write_ptr = (self.write_ptr + 1) % self.max_size
            self.current_size = min(self.current_size + 1, self.max_size)
            self.global_write_counter += 1

    def query(
        self,
        queries: torch.Tensor,
        k: int = 5,
        query_regimes: Optional[torch.Tensor] = None,
        regime_mismatch_lambda: float = 0.0,
        clip_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euclidean k-NN lookup with configurable kernel + temporal decay.

        Optional: if query_regimes are provided and regime_mismatch_lambda > 0,
        adds a constant distance penalty to memory entries whose regime differs
        from the query's regime.

        Args:
            queries: [batch, embed_dim]
            k: number of neighbors
            query_regimes: optional [batch] long tensor of query regime ids
            regime_mismatch_lambda: non-negative float

        Returns:
            corrections: [batch] — weighted average of neighbor errors
        """
        batch_size = queries.size(0)
        device = queries.device

        if self.current_size == 0 or self.keys is None:
            return torch.zeros(batch_size, device=device)

        n_mem = self.current_size
        mem_keys = self.keys[:n_mem]
        mem_values = self.values[:n_mem]

        # Euclidean distances
        dists = torch.cdist(queries, mem_keys)  # [batch, n_mem]

        # Optional: regime mismatch distance penalty (Tier2 prior)
        lam = float(regime_mismatch_lambda)
        if lam > 0.0 and query_regimes is not None and self.regimes is not None:
            try:
                qreg = query_regimes.to(device=device, dtype=torch.long).view(-1, 1)  # [B,1]
                mreg = self.regimes[:n_mem].to(device=device).view(1, -1)  # [1,N]
                mismatch = (qreg != mreg).float()
                dists = dists + lam * mismatch
            except Exception:
                pass

        # Top-k nearest
        k_actual = min(k, n_mem)
        topk_dists, topk_idx = torch.topk(dists, k_actual, dim=-1, largest=False)

        # Weighting kernel
        if self.knn_kernel == 'gaussian':
            # median heuristic bandwidth per query
            median_d = topk_dists.median(dim=-1, keepdim=True).values.clamp(min=1e-6)
            weights = torch.exp(-topk_dists.square() / (2.0 * median_d.square()))
        elif self.knn_kernel == 'softmax':
            # attention-style readout over neighbors
            T = max(1e-6, float(self.knn_softmax_temp))
            weights = torch.softmax(-topk_dists / T, dim=-1)
        else:
            # default: inverse-distance
            weights = 1.0 / (topk_dists + 1e-6)

        # Apply temporal decay: weight *= decay^(age)
        if self.temporal_decay < 1.0:
            ages = self.global_write_counter - self.timestamps[:n_mem]  # [n_mem]
            topk_ages = ages[topk_idx]  # [batch, k]
            decay_weights = self.temporal_decay ** topk_ages
            weights = weights * decay_weights

        # Optional: same-regime neighbor reweighting (post-selection)
        gamma = float(self.regime_reweight_gamma)
        if gamma != 0.0 and query_regimes is not None and self.regimes is not None:
            try:
                qreg = query_regimes.to(device=device, dtype=torch.long).view(-1, 1)
                nreg = self.regimes[:n_mem][topk_idx].to(device=device)
                same = (qreg == nreg).float()
                weights = weights * (1.0 + gamma * same)
            except Exception:
                pass

        # Gather neighbor values
        topk_values = mem_values[topk_idx].squeeze(-1)  # [batch, k]

        # --- Aggregation ---
        if self.knn_agg == 'trimmed' and k_actual >= 4:
            q = float(self.knn_trim_q)
            n_trim = max(1, int(k_actual * q))
            # mask out extremes based on neighbor values
            sorted_idx = topk_values.argsort(dim=-1)
            mask = torch.ones_like(weights, dtype=torch.bool)
            if n_trim * 2 < k_actual:
                mask.scatter_(1, sorted_idx[:, :n_trim], False)
                mask.scatter_(1, sorted_idx[:, -n_trim:], False)
            weights = weights * mask.float()
        elif self.knn_agg == 'clip':
            # winsorize based on weighted center +/- clip * (weighted MAD)
            w_norm = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            center = (w_norm * topk_values).sum(dim=-1, keepdim=True)
            mad = (w_norm * (topk_values - center).abs()).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            if clip_override is not None:
                try:
                    clip_v = clip_override.to(device=device, dtype=center.dtype).view(-1, 1)
                except Exception:
                    clip_v = float(self.knn_clip)
            else:
                clip_v = float(self.knn_clip)
            lo = center - clip_v * mad
            hi = center + clip_v * mad
            topk_values = topk_values.clamp(min=lo, max=hi)

        # Normalize weights and compute correction
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        correction = (weights * topk_values).sum(dim=-1)  # [batch]

        return correction

    def clear(self):
        """Reset buffer."""
        self.keys = None
        self.values = None
        self.timestamps = None
        self.regimes = None
        self.write_ptr = 0
        self.current_size = 0
        self.global_write_counter = 0


# =============================================================================
# Main Model: ThreeTierPFC
# =============================================================================

class ThreeTierPFC(nn.Module):
    """
    Clean three-tier PFC architecture.

    Tier 1 (Slow): LimiX frozen prior — base predictions + raw embeddings
    Tier 2 (Intermediate): K-means regime detection with per-regime bias/scale
    Tier 3 (Fast): Online euclidean k-NN error correction

    Forward: output = slow_pred + regime_scale * online_scale * knn_correction + regime_bias
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        is_regression: bool = True,
        # Slow prior config
        slow_prior_type: str = 'limix',
        slow_prior_config: Optional[Dict[str, Any]] = None,
        # Tier 2: Regime detection
        regime_k: int = 8,
        # Tier 3: Online k-NN
        online_buffer_size: int = 2048,
        online_k: int = 5,
        online_scale: float = 0.5,
        correction_space: str = 'prediction',  # 'prediction' or 'logit'
        seed_buffer_from_train: bool = False,  # pre-fill buffer with train (embedding, error) pairs
        seed_buffer_size: int = 2048,  # max samples to seed
        online_min_samples: int = 100,
        online_chunk_size: int = 64,
        temporal_decay: float = 1.0,
        # Optional: improvements found by cached sweep
        use_regime_bias: bool = True,
        use_whitening: bool = False,
        whitening_subsample: int = 50000,
        whitening_eps: float = 1e-6,
        distance_metric: str = 'pca_whiten',
        shrinkage_alpha: float = 0.1,
        # Learned projection settings (distance_metric='learned_proj')
        learned_proj_dim: int = 0,        # 0 = same as embed_dim
        learned_proj_epochs: int = 20,
        learned_proj_lr: float = 1e-3,
        learned_proj_k: int = 20,
        learned_proj_subsample: int = 20000,
        # Cross-attention readout settings
        # Modern Hopfield readout settings
        # Drift EMA correction
        use_drift_correction: bool = False,
        drift_correction_alpha: float = 0.05,
        # Adaptive online_scale settings
        use_adaptive_scale: bool = False,
        adaptive_scale_ema_alpha: float = 0.05,
        adaptive_scale_dist_w: float = 0.1,
        adaptive_scale_agree_w: float = 0.1,
        adaptive_scale_min: float = 0.3,
        adaptive_scale_max: float = 1.5,
        # Meta-correction MLP settings
        # Embedding quality calibration (auto-scale online_scale based on kNN quality)
        use_embedding_quality_scale: bool = True,
        embedding_quality_k: int = 10,
        embedding_quality_subsample: int = 10000,
        embedding_quality_min_scale: float = 0.1,
        # Online-scale calibration (val-fitted, closed-form)
        fit_global_online_scale: bool = False,
        global_online_scale_ridge: float = 1e-3,
        global_online_scale_clip: float = 2.0,
        # Per-regime online_scale gating
        use_regime_online_scale: bool = False,
        fit_regime_online_scale: bool = False,
        regime_online_scale_ridge: float = 1e-3,
        regime_online_scale_clip: float = 2.0,
        # Tier 3.5: CfC temporal correction
        # CfC online fine-tuning
        # Ensemble prior integration (Tier 2)
        use_ensemble_gating: bool = False,
        ensemble_variants: list = None,  # e.g. ["ts", "tn", "seed1", "seed2", "seed3"]
        ensemble_mode: str = 'fixed',  # 'fixed', 'adaptive' (learned MLP), or 'online' (EMA adaptation)
        ensemble_weights: dict = None,  # e.g. {"ts": 0.28, "tn": 0.05, ...} for fixed mode
        ensemble_gating_hidden: int = 128,
        ensemble_gating_lr: float = 1e-3,
        ensemble_gating_epochs: int = 80,
        ensemble_gating_wd: float = 2e-5,
        # Scheme C: Disagreement-driven correction scaling
        use_disagree_scale: bool = False,
        disagree_strength: float = 1.0,   # how much to boost correction when priors disagree
        disagree_ema_alpha: float = 0.05,  # EMA smoothing for disagreement normalization
        disagree_scale_min: float = 0.5,   # floor for the multiplier
        disagree_scale_max: float = 2.0,   # ceiling for the multiplier
        # Cascaded residual correction (Layer 2)
        use_cascaded_residual: bool = False,
        cascaded_scale: float = 0.194,
        cascaded_k: int = 47,
        cascaded_temp: float = 1.30,
        cascaded_clip: float = 2.67,
        cascaded_buf_size: int = 4096,
        cascaded_warmup: int = 385,
        # Cascaded L2 dual-buffer (Variant B)
        cascaded_use_dual_buffer: bool = False,
        cascaded_short_buf2: int = 256,
        cascaded_long_buf2: int = 4096,
        cascaded_short_temp2: float = 7.0,
        cascaded_long_temp2: float = 2.0,
        cascaded_dual_mix2: float = 0.5,
        # Device
        device: str = 'cuda',
        # Legacy params (accepted but ignored for backwards compat)
        **kwargs,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.is_regression = is_regression
        self.device = device

        # d_hidden, d_features, d_intermediate, d_fast, n_regimes and similar sizes are
        # accepted for compatibility with earlier configurations; the tiers below carry
        # their own dimensions.

        # =====================================================================
        # Tier 1: Slow Prior (LimiX)
        # =====================================================================
        # Handle 'auto' prior selection (deferred until fit() when we know d_in/n_train)
        self._auto_prior = (slow_prior_type == 'auto')
        if self._auto_prior:
            # Resolved at fit() time, once the dataset characteristics are known
            slow_prior_type = 'limix'
            print("Adaptive prior selection enabled — will resolve at fit() time")

        self.slow_prior_type = slow_prior_type
        self.slow_prior_config = slow_prior_config or {}

        prior_kwargs = {
            'd_in': d_in,
            'd_out': d_out,
            'feature_dim': 128,  # Required by SlowPrior interface, but we use raw_embeddings
            'is_regression': is_regression,
            'device': device,
        }

        if slow_prior_type == 'limix':
            prior_kwargs.update({
                'model_path': self.slow_prior_config.get('model_path', None),
                'config_path': self.slow_prior_config.get('config_path', None),
                'context_size': self.slow_prior_config.get('context_size', 10000),
                'use_retrieval': self.slow_prior_config.get('use_retrieval', False),
            })
        else:
            # Pass through all slow_prior_config keys for non-LimiX priors
            prior_kwargs.update(self.slow_prior_config)

        self.slow_prior = create_slow_prior(prior_type=slow_prior_type, **prior_kwargs)

        # Optional: TabPFN auxiliary prior (does NOT replace slow_pred; used for gating)
        self.use_tabpfn_aux = bool(kwargs.pop('use_tabpfn_aux', False))
        self.tabpfn_aux_weight = float(kwargs.pop('tabpfn_aux_weight', 0.0))
        self.tabpfn_aux_clip = float(kwargs.pop('tabpfn_aux_clip', 0.2))
        self.tabpfn_aux_config = kwargs.pop('tabpfn_aux_config', None)
        self.tabpfn_aux = None
        if self.use_tabpfn_aux:
            try:
                aux_cfg = self.tabpfn_aux_config or {}
                self.tabpfn_aux = create_slow_prior(
                    prior_type='tabpfn',
                    d_in=d_in,
                    d_out=d_out,
                    feature_dim=128,
                    is_regression=is_regression,
                    device=device,
                    **aux_cfg,
                )
            except Exception as e:
                print(f"Warning: failed to init tabpfn_aux: {e}")
                self.tabpfn_aux = None

        # =====================================================================
        # Ensemble Prior Gating (Tier 2: sample-adaptive integration)
        # =====================================================================
        self.use_ensemble_gating = use_ensemble_gating
        self.ensemble_variants = ensemble_variants or []
        self.ensemble_mode = ensemble_mode  # 'fixed' or 'adaptive'
        self.ensemble_weights_config = ensemble_weights or {}  # user-specified fixed weights
        self.ensemble_gating_hidden = ensemble_gating_hidden
        self.ensemble_gating_lr = ensemble_gating_lr
        self.ensemble_gating_epochs = ensemble_gating_epochs
        self.ensemble_gating_wd = ensemble_gating_wd
        self._ensemble_gating = None  # initialized after prior caches loaded (adaptive mode)
        self._ensemble_fixed_weights = None  # [n_priors] tensor (fixed mode)
        self._ensemble_prior_caches = {}  # variant_name -> {split -> np.array}
        self._ensemble_split_available = {}  # split -> bool tensor over prior columns
        self._ensemble_missing_warned = set()
        self._ensemble_prior_names = []  # ordered list: ["idx", "ts", "tn", ...]

        # Scheme C: Disagreement-driven correction scaling
        self.use_disagree_scale = use_disagree_scale
        self.disagree_strength = float(disagree_strength)
        self.disagree_ema_alpha = float(disagree_ema_alpha)
        self.disagree_scale_min = float(disagree_scale_min)
        self.disagree_scale_max = float(disagree_scale_max)
        self._disagree_ema_mean = None  # running mean of disagreement
        self._disagree_ema_std = None   # running std of disagreement

        # Cascaded residual correction (Layer 2)
        self.use_cascaded_residual = use_cascaded_residual
        self.cascaded_scale = float(cascaded_scale)
        self.cascaded_k = int(cascaded_k)
        self.cascaded_temp = float(cascaded_temp)
        self.cascaded_clip = float(cascaded_clip)
        self.cascaded_buf_size = int(cascaded_buf_size)
        self.cascaded_warmup = int(cascaded_warmup)
        self._cascaded_buffer = None  # initialized lazily
        self._cascaded_samples_seen = 0
        # Dual-buffer L2 (Variant B)
        self.cascaded_use_dual_buffer = cascaded_use_dual_buffer
        self.cascaded_short_buf2 = int(cascaded_short_buf2)
        self.cascaded_long_buf2 = int(cascaded_long_buf2)
        self.cascaded_short_temp2 = float(cascaded_short_temp2)
        self.cascaded_long_temp2 = float(cascaded_long_temp2)
        self.cascaded_dual_mix2 = float(cascaded_dual_mix2)
        self._cascaded_short_buffer = None  # lazy init
        self._cascaded_long_buffer = None   # lazy init
        # Adaptive L2 gate: track whether L2 helps, auto-disable if it hurts
        self._cascaded_adaptive = True  # enable adaptive gate by default
        self._cascaded_ema_benefit = 0.0  # EMA of (err_without² - err_with²), positive = L2 helps
        self._cascaded_ema_alpha = 0.002  # EMA smoothing factor (slower, more stable)
        self._cascaded_eval_after = 2000  # only evaluate after seeing enough samples
        self._cascaded_disable_threshold = -1e-5  # must be consistently negative to disable
        self._cascaded_l2_applied_count = 0
        self._cascaded_disabled = False  # permanently disabled if adaptive gate triggers
        self._cascaded_benefit_history = []  # track batch benefits for robust evaluation
        self._label_std = 1.0  # label std for denormalized gate metric (set externally via set_label_std)

        # Store for external access
        self._teacher_pred = None
        self._last_raw_embeddings = None

        # =====================================================================
        # Tier 2: Regime Detector (k-means, fitted during bio_learn)
        # =====================================================================
        self.regime_detector = RegimeDetector(n_regimes=regime_k, device=device)
        # Optional subsampling for regime k-means fit (CPU bottleneck)
        self.regime_fit_subsample = kwargs.pop('regime_fit_subsample', None)

        # =====================================================================
        # Tier 3: Online k-NN Buffer(s)
        # =====================================================================
        self.online_buffer = OnlineKNNBuffer(
            max_size=online_buffer_size,
            device=device,
            temporal_decay=temporal_decay,
            knn_kernel=kwargs.pop('knn_kernel', 'inverse'),
            knn_agg=kwargs.pop('knn_agg', 'mean'),
            knn_trim_q=kwargs.pop('knn_trim_q', 0.1),
            knn_clip=kwargs.pop('knn_clip', 3.0),
            knn_softmax_temp=kwargs.pop('knn_softmax_temp', 2.0),
            regime_reweight_gamma=float(kwargs.pop('regime_reweight_gamma_long', 0.0)),
        )

        # Initialize Hopfield readout after buffer (needs knn_agg/knn_clip)

        # Optional dual-timescale memory: blend a short-term buffer with the long-term buffer
        self.use_dual_buffer = bool(kwargs.pop('use_dual_buffer', False))
        self.dual_buffer_size = int(kwargs.pop('dual_buffer_size', 512))
        self.dual_buffer_mix_w = float(kwargs.pop('dual_buffer_mix_w', 0.2))

        # Optional: learn a per-sample gate for mixing short vs long kNN corrections
        self.use_mixw_gate = bool(kwargs.pop('use_mixw_gate', False))
        self.mixw_gate_path = kwargs.pop('mixw_gate_path', None)
        self.mixw_gate_feat_dim = int(kwargs.pop('mixw_gate_feat_dim', 5))
        self.mixw_gate = None
        if self.use_mixw_gate:
            self.mixw_gate = nn.Sequential(
                nn.Linear(self.mixw_gate_feat_dim, 1),
                nn.Sigmoid(),
            )
            if self.mixw_gate_path is not None and os.path.exists(str(self.mixw_gate_path)):
                try:
                    sd = torch.load(str(self.mixw_gate_path), map_location=device)
                    self.mixw_gate.load_state_dict(sd)
                    print(f"Loaded mixw_gate from {self.mixw_gate_path}")
                except Exception as e:
                    print(f"Warning: failed to load mixw_gate from {self.mixw_gate_path}: {e}")
        # Optional: separate temporal decays for long vs short memory
        self.dual_buffer_long_decay = float(kwargs.pop('dual_buffer_long_decay', temporal_decay))
        self.dual_buffer_short_decay = float(kwargs.pop('dual_buffer_short_decay', temporal_decay))

        # Optional: route short-term memory by intermediate regimes (Tier2 as router)
        self.use_regime_short_buffers = bool(kwargs.pop('use_regime_short_buffers', False))
        self.regime_short_mode = str(kwargs.pop('regime_short_mode', 'hard'))  # hard | soft2
        self.regime_short_tau = float(kwargs.pop('regime_short_tau', 1.0))

        # Optional: use Tier2 regimes as a soft prior in kNN lookup (do not route buffers)
        self.use_regime_knn_penalty = bool(kwargs.pop('use_regime_knn_penalty', False))
        self.regime_knn_lambda_long = float(kwargs.pop('regime_knn_lambda_long', 0.0))
        self.regime_knn_lambda_short = float(kwargs.pop('regime_knn_lambda_short', 0.0))

        # Optional: regime-confidence adaptive temperature (Proposal 1)
        self.use_regime_conf_temp = bool(kwargs.pop('use_regime_conf_temp', False))
        self.regime_conf_temp_beta_long = float(kwargs.pop('regime_conf_temp_beta_long', 0.0))
        self.regime_conf_temp_beta_short = float(kwargs.pop('regime_conf_temp_beta_short', 0.0))

        # Optional: regime-conditioned clipping (Proposal 3)
        self.use_regime_clip = bool(kwargs.pop('use_regime_clip', False))
        self.regime_clip_min = float(kwargs.pop('regime_clip_min', 0.8))
        self.regime_clip_max = float(kwargs.pop('regime_clip_max', 3.0))

        # Optional: drift-based scheduling (Tier2 as drift detector)
        self.use_drift_mixw = bool(kwargs.pop('use_drift_mixw', False))
        self.use_drift_k = bool(kwargs.pop('use_drift_k', False))
        self.drift_ema_alpha = float(kwargs.pop('drift_ema_alpha', 0.01))
        self.drift_threshold = float(kwargs.pop('drift_threshold', 0.25))
        # mix_w scheduling
        self.drift_mixw_low = float(kwargs.pop('drift_mixw_low', self.dual_buffer_mix_w))
        self.drift_mixw_high = float(kwargs.pop('drift_mixw_high', min(0.8, self.dual_buffer_mix_w + 0.1)))
        # k scheduling
        self.drift_k_low = int(kwargs.pop('drift_k_low', online_k))
        self.drift_k_high = int(kwargs.pop('drift_k_high', max(online_k, 2 * online_k)))

        # Optional: short/long buffer can have different softmax temperatures
        self.dual_buffer_short_softmax_temp = float(kwargs.pop('dual_buffer_short_softmax_temp', self.online_buffer.knn_softmax_temp))
        self.dual_buffer_long_softmax_temp = float(kwargs.pop('dual_buffer_long_softmax_temp', self.online_buffer.knn_softmax_temp))

        # Optional: short buffer can use raw (unwhitened) embeddings to preserve local geometry
        # while long buffer uses whitened embeddings.
        self.dual_buffer_short_use_raw = bool(kwargs.pop('dual_buffer_short_use_raw', False))

        self._drift_ema_abs_err = 0.0

        # Enhanced drift-aware buffer: smooth sigmoid mechanism (replaces binary threshold)
        self.drift_base_mix_w = float(kwargs.pop('drift_base_mix_w', self.dual_buffer_mix_w))
        self.drift_alpha = float(kwargs.pop('drift_alpha', 3.0))
        self.drift_window_size = int(kwargs.pop('drift_window_size', 256))
        # Rolling error window for drift signal computation
        self._drift_err_window = []  # list of floats, max length = drift_window_size

        # Online Embedding Adaptation: low-rank projection e_adapted = e @ (I + UV^T)
        self.use_adaptive_embedding = bool(kwargs.pop('use_adaptive_embedding', False))
        self.adaptive_rank = int(kwargs.pop('adaptive_rank', 4))
        self.adaptive_lr = float(kwargs.pop('adaptive_lr', 1e-4))
        self.adaptive_weight_decay = float(kwargs.pop('adaptive_weight_decay', 1e-4))
        # U, V matrices initialized lazily (need embed_dim from first forward pass)
        self._adaptive_U = None  # (D, rank)
        self._adaptive_V = None  # (D, rank)
        self._adaptive_embed_dim = None
        self._adaptive_pre_corr = None
        self._adaptive_unsupported_warned = False

        # Optional: surprise-driven short buffer writes (independent component)
        # - 'all': write all samples (default, preserves current behavior)
        # - 'topm': per update, only write the top-m by |error|
        # - 'quantile': write samples with |error| >= quantile(|error|)
        # - 'dedup': only write if nearest short key distance >= eps
        self.short_write_strategy = str(kwargs.pop('short_write_strategy', 'all'))
        self.short_write_topm = int(kwargs.pop('short_write_topm', 16))
        self.short_write_quantile = float(kwargs.pop('short_write_quantile', 0.9))
        self.short_write_dedup_eps = float(kwargs.pop('short_write_dedup_eps', 0.1))

        self.dual_buffer_short_k = int(kwargs.pop('dual_buffer_short_k', online_k))
        self.dual_buffer_long_k = int(kwargs.pop('dual_buffer_long_k', online_k))

        # Apply long decay to the long-term buffer
        self.online_buffer.temporal_decay = self.dual_buffer_long_decay

        self.short_buffer = OnlineKNNBuffer(
            max_size=self.dual_buffer_size,
            device=device,
            temporal_decay=self.dual_buffer_short_decay,
            knn_kernel=self.online_buffer.knn_kernel,
            knn_agg=self.online_buffer.knn_agg,
            knn_trim_q=self.online_buffer.knn_trim_q,
            knn_clip=self.online_buffer.knn_clip,
            knn_softmax_temp=self.online_buffer.knn_softmax_temp,
            regime_reweight_gamma=float(kwargs.pop('regime_reweight_gamma_short', 0.0)),
        ) if self.use_dual_buffer else None

        # Per-regime short buffers (created lazily after regime_k is known)
        self.regime_short_buffers = None
        if self.use_dual_buffer and self.use_regime_short_buffers:
            self.regime_short_buffers = [
                OnlineKNNBuffer(
                    max_size=self.dual_buffer_size,
                    device=device,
                    temporal_decay=self.dual_buffer_short_decay,
                    knn_kernel=self.online_buffer.knn_kernel,
                    knn_agg=self.online_buffer.knn_agg,
                    knn_trim_q=self.online_buffer.knn_trim_q,
                    knn_clip=self.online_buffer.knn_clip,
                    knn_softmax_temp=self.online_buffer.knn_softmax_temp,
                    regime_reweight_gamma=float(kwargs.get('regime_reweight_gamma_short', 0.0)),
                )
                for _ in range(regime_k)
            ]

        self.online_k = online_k
        self.online_scale = online_scale
        self.correction_space = str(correction_space)  # 'prediction' or 'logit'
        self.seed_buffer_from_train = seed_buffer_from_train
        self.seed_buffer_size = int(seed_buffer_size)
        self.online_min_samples = online_min_samples
        self._train_embeddings_for_seed = None  # stored after bio-learning
        self._train_errors_for_seed = None
        self.online_chunk_size = online_chunk_size
        self.online_mode = False

        # Sweep-found flags
        self.use_regime_bias = use_regime_bias
        self.use_whitening = use_whitening
        self.whitening_subsample = int(whitening_subsample)
        self.whitening_eps = float(whitening_eps)
        self.distance_metric = distance_metric  # 'pca_whiten' | 'shrinkage_mahal' | 'learned_proj'
        self.shrinkage_alpha = float(shrinkage_alpha)
        self.learned_proj_dim = int(learned_proj_dim)
        self.learned_proj_epochs = int(learned_proj_epochs)
        self.learned_proj_lr = float(learned_proj_lr)
        self.learned_proj_k = int(learned_proj_k)
        self.learned_proj_subsample = int(learned_proj_subsample)

        # Cross-attention readout

        # Meta-correction MLP
        # Drift correction
        self.use_drift_correction = use_drift_correction
        self.drift_correction_alpha = float(drift_correction_alpha)

        # Adaptive scale
        self.use_adaptive_scale = use_adaptive_scale
        self.adaptive_scale_ema_alpha = float(adaptive_scale_ema_alpha)
        self.adaptive_scale_dist_w = float(adaptive_scale_dist_w)
        self.adaptive_scale_agree_w = float(adaptive_scale_agree_w)
        self.adaptive_scale_min = float(adaptive_scale_min)
        self.adaptive_scale_max = float(adaptive_scale_max)


        # Embedding quality calibration
        self.use_embedding_quality_scale = use_embedding_quality_scale
        self.embedding_quality_k = int(embedding_quality_k)
        self.embedding_quality_subsample = int(embedding_quality_subsample)
        self.embedding_quality_min_scale = float(embedding_quality_min_scale)
        self._embedding_quality_mult = 1.0  # Set in finalize_bio_learning

        # Online-scale calibration (global scalar multiplier)
        self.fit_global_online_scale = fit_global_online_scale
        self.global_online_scale_ridge = float(global_online_scale_ridge)
        self.global_online_scale_clip = float(global_online_scale_clip)
        self.online_scale_mult = 1.0

        # Per-regime online_scale gating
        self.use_regime_online_scale = use_regime_online_scale
        self.fit_regime_online_scale = fit_regime_online_scale
        self.regime_online_scale_ridge = regime_online_scale_ridge
        self.regime_online_scale_clip = regime_online_scale_clip

        # Tree leaf embeddings for classification kNN (pre-computed, loaded from disk)
        self.cls_embedding_type = str(kwargs.pop('cls_embedding_type', 'limix'))
        self._tree_leaf_embeddings = {}  # split -> np.ndarray [n_samples, 64]

        # Whitening transform (computed in finalize_bio_learning)
        self._whiten_mu = None    # [embed_dim]
        self._whiten_W = None     # [embed_dim, embed_dim]

        # =====================================================================
        # Tier 3.5: CfC Temporal Correction (optional)
        # =====================================================================


        # Track step count for compatibility
        self.step_count = 0

    # =========================================================================
    # Slow prior management (delegated)
    # =========================================================================

    def set_slow_prior_context(self, X: np.ndarray, y: np.ndarray):
        """Set training context for slow prior."""
        # Resolve 'auto' prior selection now that we know dataset characteristics
        if self._auto_prior:
            try:
                from model.lib.adaptive_prior_selector import select_prior
            except ImportError as exc:  # not part of this release
                raise ImportError(
                    "slow_prior_type='auto' needs model/lib/adaptive_prior_selector.py, "
                    "which is not part of this repository. Name the prior explicitly; "
                    "configs/default/three_tier_pfc_bio.json sets slow_prior_type='limix'."
                ) from exc
            n_features = X.shape[1]
            n_train = X.shape[0]
            task = "regression" if self.is_regression else "binclass"
            selected_type, config_overrides = select_prior(n_features, n_train, task)

            if selected_type != self.slow_prior_type:
                from model.lib.adaptive_prior_selector import get_prior_rationale
                print(f"[AutoPrior] {get_prior_rationale(selected_type, n_features, n_train)}")
                self.slow_prior_type = selected_type
                self.slow_prior_config.update(config_overrides)

                # Rebuild slow prior
                prior_kwargs = {
                    'd_in': self.d_in,
                    'd_out': self.d_out,
                    'feature_dim': 128,
                    'is_regression': self.is_regression,
                    'device': str(self.device) if hasattr(self, 'device') else 'cuda',
                }
                if selected_type == 'limix':
                    prior_kwargs.update({
                        'model_path': self.slow_prior_config.get('model_path', None),
                        'config_path': self.slow_prior_config.get('config_path', None),
                        'context_size': self.slow_prior_config.get('context_size', 10000),
                        'use_retrieval': self.slow_prior_config.get('use_retrieval', False),
                    })
                else:
                    prior_kwargs.update(self.slow_prior_config)

                from model.lib.slow_priors import create_slow_prior
                self.slow_prior = create_slow_prior(prior_type=selected_type, **prior_kwargs)
                print(f"[AutoPrior] Rebuilt slow prior as '{selected_type}'")

            self._auto_prior = False  # Resolved

        self.slow_prior.set_context(X, y)
        if self.use_tabpfn_aux and self.tabpfn_aux is not None:
            self.tabpfn_aux.set_context(X, y)

    def precompute_slow_prior(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        split: str,
        cache_dir: str = None,
        dataset_name: str = None,
    ):
        """Pre-compute slow prior predictions/embeddings for caching."""
        import inspect
        sig = inspect.signature(self.slow_prior.precompute_predictions)
        if 'dataset_name' in sig.parameters:
            self.slow_prior.precompute_predictions(
                X, indices, split, cache_dir=cache_dir, dataset_name=dataset_name
            )
        else:
            self.slow_prior.precompute_predictions(X, indices, split, cache_dir=cache_dir)

        # Also precompute TabPFN auxiliary predictions if enabled
        if self.use_tabpfn_aux and self.tabpfn_aux is not None:
            sig2 = inspect.signature(self.tabpfn_aux.precompute_predictions)
            if 'dataset_name' in sig2.parameters:
                self.tabpfn_aux.precompute_predictions(
                    X, indices, split, cache_dir=cache_dir, dataset_name=dataset_name
                )
            else:
                self.tabpfn_aux.precompute_predictions(X, indices, split, cache_dir=cache_dir)

    def set_raw_target_stats(self, mean: float, std: float):
        """Forward the pipeline's label statistics to the prior and keep them for the
        ensemble variant caches, which are written by the same offline builders."""
        self._raw_target_mean = float(mean)
        self._raw_target_std = float(std) + 1e-8
        if hasattr(self.slow_prior, 'set_raw_target_stats'):
            self.slow_prior.set_raw_target_stats(mean, std)

    def _align_raw_cache(self, arr):
        """Convert a variant cache in raw target units to the standardised space."""
        m = getattr(self, '_raw_target_mean', None)
        if m is None or not self.is_regression or arr.size == 0:
            return arr
        s = self._raw_target_std
        a = np.asarray(arr, dtype=np.float64)
        if abs(float(a.mean()) - m) / s >= 1.0:
            return arr
        return ((a - m) / s).astype(np.float32)

    def load_ensemble_prior_caches(self, dataset_name: str, ctx: int):
        """Load variant prior caches for ensemble gating.

        Expects caches at: cache/limix/{dataset_name}-{variant}/limix_predictions_{split}_ctx{ctx}.npy
        Base (idx) prior is already loaded via self.slow_prior.

        Args:
            dataset_name: e.g. "cooking-time", "weather"
            ctx: context size used for LimiX caching
        """
        if not self.use_ensemble_gating or not self.ensemble_variants:
            return

        self._ensemble_prior_names = ["idx"]  # base prior always first

        # Get base prior stats for rescaling
        base_train_path = f"cache/limix/{dataset_name}/limix_predictions_train_ctx{ctx}.npy"
        if os.path.exists(base_train_path):
            base_train = self._align_raw_cache(np.load(base_train_path).astype(np.float32))
            self._ensemble_base_mean = base_train.mean()
            self._ensemble_base_std = base_train.std() + 1e-8
        else:
            self._ensemble_base_mean = 0.0
            self._ensemble_base_std = 1.0

        variant_suffix_map = {
            "ts": "ts", "tn": "tn",
            "seed1": "seed1", "seed2": "seed2", "seed3": "seed3",
            "ts-seed1": "ts-seed1", "ts-seed2": "ts-seed2",
        }

        for variant in self.ensemble_variants:
            # XGBoost prior uses a different cache path
            if variant == 'xgb':
                cache_dir = f"cache/xgb/{dataset_name}"
                file_pattern = lambda split: f"xgb_predictions_{split}.npy"
            else:
                suffix = variant_suffix_map.get(variant, variant)
                cache_dir = f"cache/limix/{dataset_name}-{suffix}"
                file_pattern = lambda split: f"limix_predictions_{split}_ctx{ctx}.npy"

            loaded_any = False
            self._ensemble_prior_caches[variant] = {}
            for split in ['train', 'val', 'test']:
                path = os.path.join(cache_dir, file_pattern(split))
                if os.path.exists(path):
                    preds = self._align_raw_cache(np.load(path).astype(np.float32))
                    # Rescale to match base prior distribution
                    preds_rescaled = (
                        (preds - preds.mean()) / (preds.std() + 1e-8)
                        * self._ensemble_base_std + self._ensemble_base_mean
                    )
                    self._ensemble_prior_caches[variant][split] = preds_rescaled
                    loaded_any = True
                    print(f"  Ensemble: loaded {variant}/{split} ({len(preds)} samples)")

            if loaded_any:
                self._ensemble_prior_names.append(variant)
            else:
                print(f"  Ensemble: WARNING - no caches found for variant '{variant}' in {cache_dir}")

        n_priors = len(self._ensemble_prior_names)
        print(f"Ensemble: {n_priors} priors ({', '.join(self._ensemble_prior_names)}), mode={self.ensemble_mode}")

        if n_priors < 2:
            print("  WARNING: fewer than 2 priors, disabling ensemble gating")
            self.use_ensemble_gating = False
            return

        self._ensemble_n_priors = n_priors

        # For fixed mode: compute weight vector from config
        if self.ensemble_mode in ('fixed', 'online'):
            weights = []
            for name in self._ensemble_prior_names:
                if name == 'idx':
                    # idx weight = 1 - sum(other weights)
                    other_sum = sum(
                        self.ensemble_weights_config.get(v, 1.0 / n_priors)
                        for v in self._ensemble_prior_names if v != 'idx'
                    )
                    weights.append(max(0.0, 1.0 - other_sum))
                else:
                    weights.append(self.ensemble_weights_config.get(name, 1.0 / n_priors))
            # Normalize
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]
            self._ensemble_fixed_weights = torch.tensor(weights, dtype=torch.float32)
            print(f"  Fixed weights: {dict(zip(self._ensemble_prior_names, weights))}")

    @staticmethod
    def _renormalise_over_available(weights: torch.Tensor, available: Optional[torch.Tensor]):
        """Zero the weights of priors with no cache for this split and rescale to sum one.

        `weights` is [B, n_priors] or [1, n_priors]; `available` is a bool mask over the
        prior columns. Rescaling keeps the weighted sum on the same scale whatever subset
        of the pool a split holds.
        """
        if available is None or bool(available.all()):
            return weights
        w = weights * available.to(weights.dtype).unsqueeze(0)
        s = w.sum(dim=1, keepdim=True)
        return torch.where(s > 0, w / s, weights)

    def _get_ensemble_predictions(self, indices: torch.Tensor, split: str) -> Optional[torch.Tensor]:
        """Get stacked predictions from all ensemble priors for given indices.

        Args:
            indices: [B] — sample indices
            split: 'train', 'val', or 'test'

        Returns:
            preds: [B, n_priors] tensor, or None if not available
        """
        if not self.use_ensemble_gating or not self._ensemble_prior_names:
            return None

        device = indices.device
        idx_np = indices.cpu().numpy().astype(int)
        B = len(idx_np)
        n_priors = len(self._ensemble_prior_names)

        preds = np.zeros((B, n_priors), dtype=np.float32)
        available = np.zeros(n_priors, dtype=bool)

        # First prior is always "idx" (base) — get from slow_prior cache
        base_cache = self.slow_prior.prediction_cache.get(split, {})
        available[0] = bool(base_cache)
        for i, idx in enumerate(idx_np):
            if int(idx) in base_cache:
                preds[i, 0] = float(base_cache[int(idx)])

        # Remaining priors from variant caches
        for p_idx, name in enumerate(self._ensemble_prior_names[1:], 1):
            variant_split = self._ensemble_prior_caches.get(name, {}).get(split, None)
            if variant_split is not None:
                available[p_idx] = True
                # variant_split is a flat numpy array indexed by position
                for i, idx in enumerate(idx_np):
                    if idx < len(variant_split):
                        preds[i, p_idx] = variant_split[idx]

        # A prior with no cache for this split contributes no column, so its weight comes
        # off the blend as well and the weighted sum keeps its scale.
        self._ensemble_split_available[split] = torch.tensor(available, dtype=torch.bool)
        if not available.all() and split not in self._ensemble_missing_warned:
            self._ensemble_missing_warned.add(split)
            missing = [n for n, a in zip(self._ensemble_prior_names, available) if not a]
            kept = float(self._ensemble_fixed_weights[torch.tensor(available)].sum()) \
                if self._ensemble_fixed_weights is not None else float('nan')
            print(f"  Ensemble: {missing} have no cache for split '{split}'. Blending over "
                  f"the {int(available.sum())} priors that are present, with their weights "
                  f"renormalised; they carry {kept:.4f} of the configured weight.")

        return torch.tensor(preds, device=device, dtype=torch.float32)

    def _train_ensemble_gating(self, all_embeddings: np.ndarray, all_errors: np.ndarray):
        """Train the ensemble gating MLP on training set data.

        Uses accumulated embeddings and the per-prior prediction errors to
        learn sample-adaptive weights.

        Args:
            all_embeddings: [N, embed_dim] — raw embeddings from bio-learning
            all_errors: [N] — slow_pred errors (y_true - slow_pred)
        """
        if not self.use_ensemble_gating:
            return

        device = next(self.parameters()).device
        n_priors = self._ensemble_n_priors
        N = len(all_embeddings)
        embed_dim = all_embeddings.shape[1]

        print(f"\n{'='*60}")
        print(f"Training Ensemble Gating MLP: {n_priors} priors, {N} samples")
        print(f"{'='*60}")

        # Get Y_train for computing per-prior errors
        # We reconstruct Y from slow_pred + errors: Y = slow_pred + error
        # But we need actual per-prior predictions, not just base errors.
        # The priors are stored as prediction values, not errors.

        # Get base prior predictions for train split
        base_cache = self.slow_prior.prediction_cache.get('train', {})
        base_preds = np.array([float(base_cache.get(i, 0.0)) for i in range(N)], dtype=np.float32)

        # Y_train = base_preds + all_errors (since error = y_true - slow_pred)
        Y_train = base_preds + all_errors

        # Stack all prior predictions [N, n_priors]
        all_prior_preds = np.zeros((N, n_priors), dtype=np.float32)
        all_prior_preds[:, 0] = base_preds

        for p_idx, name in enumerate(self._ensemble_prior_names[1:], 1):
            variant_train = self._ensemble_prior_caches.get(name, {}).get('train', None)
            if variant_train is not None:
                n_avail = min(N, len(variant_train))
                all_prior_preds[:n_avail, p_idx] = variant_train[:n_avail]
            else:
                # If no train cache for this variant, use base predictions
                all_prior_preds[:, p_idx] = base_preds
                print(f"  WARNING: no train cache for '{name}', using base predictions")

        # Whiten embeddings (use the whitening already computed)
        if self._whiten_mu is not None and self._whiten_W is not None:
            emb_w = (all_embeddings - self._whiten_mu.cpu().numpy()) @ self._whiten_W.cpu().numpy()
        else:
            # Compute whitening locally
            mu = all_embeddings.mean(axis=0)
            c = all_embeddings - mu
            cov = (c.T @ c) / max(N - 1, 1) + 1e-5 * np.eye(embed_dim)
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx_sort = np.argsort(eigvals)[::-1]
            W = eigvecs[:, idx_sort] * (1.0 / np.sqrt(eigvals[idx_sort] + 1e-10))
            emb_w = (all_embeddings - mu) @ W

        # Create and train gating MLP
        gating = EnsemblePriorGating(emb_w.shape[1], n_priors, self.ensemble_gating_hidden).to(device)
        optimizer = torch.optim.Adam(gating.parameters(), lr=self.ensemble_gating_lr,
                                     weight_decay=self.ensemble_gating_wd)

        emb_t = torch.tensor(emb_w, dtype=torch.float32, device=device)
        preds_t = torch.tensor(all_prior_preds, dtype=torch.float32, device=device)
        y_t = torch.tensor(Y_train, dtype=torch.float32, device=device)

        batch_size = 2048
        best_loss = float('inf')
        best_state = None

        for epoch in range(self.ensemble_gating_epochs):
            perm = torch.randperm(N, device=device)
            total_loss = 0.0
            for i in range(0, N, batch_size):
                bidx = perm[i:i + batch_size]
                weights = gating(emb_t[bidx])  # [B, n_priors]
                blend = (weights * preds_t[bidx]).sum(dim=1)  # [B]
                loss = ((blend - y_t[bidx]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bidx)
            rmse = np.sqrt(total_loss / N)
            if rmse < best_loss:
                best_loss = rmse
                best_state = {k: v.clone() for k, v in gating.state_dict().items()}

        gating.load_state_dict(best_state)
        self._ensemble_gating = gating
        self._ensemble_gating.eval()

        # Report weight statistics
        with torch.no_grad():
            test_weights = gating(emb_t).cpu().numpy()
        print(f"  Training RMSE: {best_loss:.6f}")
        print(f"  Learned weight stats:")
        for i, name in enumerate(self._ensemble_prior_names):
            w = test_weights[:, i]
            print(f"    {name}: mean={w.mean():.4f}, std={w.std():.4f}")

    def load_tree_leaf_embeddings(self, dataset_name: str):
        """Load pre-computed tree leaf embeddings from cache/tree_leaf/{dataset}/."""
        if self.cls_embedding_type != 'tree_leaf' or self.is_regression:
            return
        base_dir = f"cache/tree_leaf/{dataset_name}"
        for split in ['train', 'val', 'test']:
            path = f"{base_dir}/leaf_embeddings_{split}.npy"
            if os.path.exists(path):
                self._tree_leaf_embeddings[split] = np.load(path).astype(np.float32)
                print(f"Loaded tree leaf embeddings for {split}: {self._tree_leaf_embeddings[split].shape}")
            else:
                print(f"WARNING: tree leaf embeddings not found: {path}")

    def _get_tree_leaf_emb(self, indices: torch.Tensor, split: str) -> Optional[torch.Tensor]:
        """Get tree leaf embeddings for given indices and split, or None if not available."""
        if self.cls_embedding_type != 'tree_leaf' or self.is_regression:
            return None
        emb = self._tree_leaf_embeddings.get(split, None)
        if emb is None:
            return None
        idx = indices.cpu().numpy().astype(int)
        batch_emb = emb[idx]
        return torch.tensor(batch_emb, dtype=torch.float32, device=indices.device)

    def freeze_slow_prior(self):
        """Freeze the slow prior parameters."""
        self.slow_prior.freeze()
        print("Slow prior frozen.")

    # =========================================================================
    # Bio-learning: single pass to fit regime detector
    # =========================================================================

    def init_bio_learning(self):
        """Initialize accumulators for bio-learning pass."""
        self._bio_embeddings = []
        self._bio_errors = []
        print("Bio-learning initialized (accumulating embeddings + errors)")

    def accumulate_bio_statistics(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        predictions: Optional[torch.Tensor] = None,
        slow_pred: Optional[torch.Tensor] = None,
    ):
        """Accumulate embeddings and errors during bio-learning pass."""
        if labels is None or slow_pred is None:
            return

        with torch.no_grad():
            # Use raw embeddings (192-dim) if available, else projected features
            raw_emb = self._last_raw_embeddings
            if raw_emb is None:
                raw_emb = features

            labels_flat = labels.view(-1) if labels.dim() > 1 else labels
            if not self.is_regression and slow_pred.dim() > 1 and slow_pred.shape[-1] > 1:
                # Classification: use positive class prob for binary, or max prob for multiclass
                if slow_pred.shape[-1] == 2:
                    pred_flat = slow_pred[:, 1]  # P(positive class)
                else:
                    pred_flat = slow_pred.max(dim=-1).values  # max class prob
                errors = labels_flat.float() - pred_flat
            else:
                pred_flat = slow_pred.view(-1) if slow_pred.dim() > 1 else slow_pred
                errors = labels_flat - pred_flat

            self._bio_embeddings.append(raw_emb.cpu().numpy())
            self._bio_errors.append(errors.cpu().numpy())

    def _learn_projection(self, X: np.ndarray, mu: np.ndarray, W_init: np.ndarray,
                          errors: Optional[np.ndarray] = None) -> np.ndarray:
        """Learn a projection matrix P that minimizes soft-kNN LOO prediction error.

        Initialized from PCA whitening W_init, then fine-tuned so that kNN in
        P-projected space yields better error corrections.

        Args:
            X: [n, d] subsampled embeddings (already centered by mu)
            mu: [1, d] mean
            W_init: [d, d] PCA whitening matrix
            errors: [n_full] errors array (same order as full embeddings; X may be subsampled)

        Returns:
            P: [d, d_proj] learned projection matrix (numpy)
        """
        if errors is None:
            print("Learned projection: no errors provided, falling back to PCA whitening")
            return W_init

        n, d = X.shape
        # Subsample for training speed
        sub = min(n, self.learned_proj_subsample)
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=sub, replace=False)

        device = self.device
        Xc = torch.tensor((X[idx] - mu).astype(np.float32), device=device)
        errs = torch.tensor(errors[idx].astype(np.float32), device=device).view(-1)

        d_proj = self.learned_proj_dim if self.learned_proj_dim > 0 else d
        # Initialize P from W_init (possibly truncated)
        P = torch.tensor(W_init[:, :d_proj].astype(np.float32), device=device, requires_grad=True)

        optimizer = torch.optim.Adam([P], lr=self.learned_proj_lr)
        k = min(self.learned_proj_k, sub - 1)
        T = max(1e-6, float(getattr(self.online_buffer, 'knn_softmax_temp', 2.0)) if hasattr(self, 'online_buffer') else 2.0)

        print(f"Learning projection: n={sub}, d={d}→{d_proj}, k={k}, epochs={self.learned_proj_epochs}", flush=True)

        best_loss = float('inf')
        best_P = P.detach().clone()

        for ep in range(self.learned_proj_epochs):
            optimizer.zero_grad()
            # Project
            proj = Xc @ P  # [sub, d_proj]
            # Pairwise distances
            dists = torch.cdist(proj, proj)  # [sub, sub]
            # Mask self (set diagonal to large value)
            dists = dists + torch.eye(sub, device=device) * 1e12
            # Top-k nearest
            topk_dists, topk_idx = torch.topk(dists, k, dim=-1, largest=False)
            # Softmax weights
            weights = torch.softmax(-topk_dists / T, dim=-1)  # [sub, k]
            # kNN prediction: weighted average of neighbor errors
            neighbor_errs = errs[topk_idx]  # [sub, k]
            knn_pred = (weights * neighbor_errs).sum(dim=-1)  # [sub]
            # Loss: MSE between kNN prediction and actual error
            loss = (knn_pred - errs).pow(2).mean()

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_P = P.detach().clone()

            if ep % 5 == 0 or ep == self.learned_proj_epochs - 1:
                print(f"  proj epoch {ep}: loss={loss.item():.6f}", flush=True)

        print(f"Learned projection: best loss={best_loss:.6f}", flush=True)
        return best_P.cpu().numpy()

    def _compute_whitening(self, embeddings: np.ndarray, errors: Optional[np.ndarray] = None):
        """Compute whitening matrix on a subsample of embeddings.

        Supports two distance metrics (controlled by self.distance_metric):
          - 'pca_whiten': standard PCA whitening via eigendecomposition of cov
          - 'shrinkage_mahal': shrinkage covariance (Ledoit-Wolf style)
              cov_shrunk = (1 - alpha) * cov_emp + alpha * (trace(cov_emp)/d) * I
              then W = invsqrt(cov_shrunk)

        Stores:
          self._whiten_mu (torch) and self._whiten_W (torch)

        Whitening is only used for the kNN fast memory distance metric.
        """
        if not self.use_whitening:
            return

        n, d = embeddings.shape
        if n == 0:
            return

        # Subsample for speed
        if self.whitening_subsample and n > self.whitening_subsample:
            rng = np.random.RandomState(0)
            idx = rng.choice(n, size=self.whitening_subsample, replace=False)
            X = embeddings[idx]
            errs_sub = errors[idx] if errors is not None else None
        else:
            X = embeddings
            errs_sub = errors

        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu

        # Empirical covariance (d x d)
        cov = (Xc.T @ Xc) / max(1, (len(Xc) - 1))

        if self.distance_metric == 'shrinkage_mahal':
            # Shrinkage covariance: (1-alpha)*cov_emp + alpha*(trace/d)*I
            alpha = self.shrinkage_alpha
            target = np.trace(cov) / d * np.eye(d)
            cov_shrunk = (1.0 - alpha) * cov + alpha * target
            eigvals, eigvecs = np.linalg.eigh(cov_shrunk)
            eigvals = np.maximum(eigvals, 0.0)
            W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + self.whitening_eps)) @ eigvecs.T
            label = f"shrinkage Mahalanobis (alpha={alpha})"
        elif self.distance_metric == 'learned_proj':
            # Initialize with PCA whitening, then optimize via soft-kNN LOO loss
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 0.0)
            W_init = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + self.whitening_eps)) @ eigvecs.T
            W = self._learn_projection(X, mu, W_init, errs_sub)
            label = f"learned projection (epochs={self.learned_proj_epochs})"
        else:
            # Default: PCA whitening
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 0.0)
            W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + self.whitening_eps)) @ eigvecs.T
            label = "PCA whitening"

        device = self.device
        self._whiten_mu = torch.tensor(mu.squeeze(0), dtype=torch.float32, device=device)
        self._whiten_W = torch.tensor(W, dtype=torch.float32, device=device)
        print(f"Whitening: computed {label} on n={len(X)} d={d}")

    def _whiten(self, emb: torch.Tensor) -> torch.Tensor:
        if (not self.use_whitening) or self._whiten_mu is None or self._whiten_W is None:
            return emb
        # Dimension mismatch is a bug — log warning but handle gracefully
        whiten_dim = self._whiten_mu.size(0)
        embed_dim = emb.size(-1)
        if embed_dim != whiten_dim:
            print(f"WARNING: _whiten dim mismatch: emb={embed_dim} vs whiten={whiten_dim}")
            if embed_dim > whiten_dim:
                emb = emb[..., :whiten_dim]
            else:
                pad = torch.zeros(*emb.shape[:-1], whiten_dim - embed_dim,
                                  device=emb.device, dtype=emb.dtype)
                emb = torch.cat([emb, pad], dim=-1)
        return (emb - self._whiten_mu) @ self._whiten_W



    def finalize_bio_learning(self):
        """Fit regime detector on accumulated training data.

        Returns:
            (all_embeddings, all_errors) as numpy arrays for CfC training,
            or None if no data was accumulated.
        """
        if not self._bio_embeddings:
            print("Warning: no bio-learning data accumulated")
            return None

        all_embeddings = np.concatenate(self._bio_embeddings, axis=0)
        all_errors = np.concatenate(self._bio_errors, axis=0)

        print(f"Fitting regime detector on {len(all_embeddings)} samples...")
        print(f"  Embedding dim: {all_embeddings.shape[1]}")
        print(f"  Mean error: {all_errors.mean():.4f}, Std: {all_errors.std():.4f}")

        # Optionally subsample k-means fit for speed
        subsample = getattr(self, 'regime_fit_subsample', None)
        self.regime_detector.fit(all_embeddings, all_errors, subsample=subsample)

        # Whitening for kNN distance metric (fast tier)
        self._compute_whitening(all_embeddings, all_errors)

        # Train cross-attention readout if enabled

        # Train meta-correction MLP if enabled

        # Train ensemble gating MLP if enabled (adaptive mode only)
        if self.use_ensemble_gating and self.ensemble_mode == 'adaptive':
            self._train_ensemble_gating(all_embeddings, all_errors)

        # Embedding quality calibration: measure kNN prediction quality
        if self.use_embedding_quality_scale:
            self._calibrate_embedding_quality(all_embeddings, all_errors)

        # Store train data for buffer seeding (subsample if needed)
        if self.seed_buffer_from_train:
            n = len(all_embeddings)
            if n > self.seed_buffer_size:
                rng = np.random.RandomState(42)
                idx = rng.choice(n, self.seed_buffer_size, replace=False)
                self._train_embeddings_for_seed = all_embeddings[idx]
                self._train_errors_for_seed = all_errors[idx]
            else:
                self._train_embeddings_for_seed = all_embeddings.copy()
                self._train_errors_for_seed = all_errors.copy()
            print(f"[BufferSeed] Stored {len(self._train_embeddings_for_seed)} train samples for buffer seeding")

        # Cleanup accumulators but return data for CfC training
        del self._bio_embeddings
        del self._bio_errors

        return all_embeddings, all_errors

    def _calibrate_embedding_quality(self, embeddings: np.ndarray, errors: np.ndarray):
        """Measure embedding quality via leave-one-out kNN correlation.

        If embeddings form a meaningful space for error prediction, kNN estimates
        will correlate with actual errors. If not, reduce online_scale to avoid
        hurting the slow prior's predictions.

        Sets self._embedding_quality_mult in [embedding_quality_min_scale, 1.0].
        """
        n = len(embeddings)
        k = min(self.embedding_quality_k, n - 1)
        if n < k + 1:
            print(f"[EmbQuality] Too few samples ({n}), skipping calibration")
            return

        # Subsample for speed
        if n > self.embedding_quality_subsample:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, self.embedding_quality_subsample, replace=False)
            emb_sub = embeddings[idx]
            err_sub = errors[idx]
        else:
            emb_sub = embeddings
            err_sub = errors

        n_sub = len(emb_sub)

        # Compute pairwise distances (chunked for memory)
        from scipy.spatial.distance import cdist
        chunk = 2000
        knn_preds = np.zeros(n_sub)
        for i in range(0, n_sub, chunk):
            end = min(i + chunk, n_sub)
            dists = cdist(emb_sub[i:end], emb_sub, metric='euclidean')  # [chunk, n_sub]
            # Mask self-distances
            for j in range(end - i):
                dists[j, i + j] = np.inf
            # Top-k nearest
            topk_idx = np.argpartition(dists, k, axis=1)[:, :k]
            topk_vals = err_sub[topk_idx]  # [chunk, k]
            knn_preds[i:end] = topk_vals.mean(axis=1)

        # Correlation between kNN predictions and actual errors
        corr = np.corrcoef(knn_preds, err_sub)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        # Map correlation to scale multiplier:
        # corr >= 0.3 -> mult = 1.0 (full correction)
        # corr <= 0.05 -> mult = min_scale (minimal correction)
        # Linear interpolation between
        high_thresh = 0.3
        low_thresh = 0.05
        if corr >= high_thresh:
            mult = 1.0
        elif corr <= low_thresh:
            mult = self.embedding_quality_min_scale
        else:
            frac = (corr - low_thresh) / (high_thresh - low_thresh)
            mult = self.embedding_quality_min_scale + frac * (1.0 - self.embedding_quality_min_scale)

        self._embedding_quality_mult = float(mult)
        print(f"[EmbQuality] kNN-error correlation: {corr:.4f} → online_scale multiplier: {mult:.3f}")

    # =========================================================================
    # CfC Training
    # =========================================================================


    # =========================================================================
    # Online adaptation control
    # =========================================================================

    def _seed_buffer_from_training(self):
        """Pre-fill online buffer with training (embedding, error) pairs.

        This gives kNN correction meaningful neighbors from the start,
        especially important for imbalanced classification where test-time
        errors are sparse (e.g., 3.5% fraud rate → most test errors ≈ 0).
        """
        import torch
        emb = self._train_embeddings_for_seed
        err = self._train_errors_for_seed
        if emb is None or err is None:
            return

        device = next(self.parameters()).device

        # Whiten embeddings (same transform as used for kNN queries)
        emb_t = torch.tensor(emb, dtype=torch.float32, device=device)
        emb_whitened = self._whiten(emb_t)

        err_t = torch.tensor(err, dtype=torch.float32, device=device)

        # Assign regimes for buffer entries
        regime_ids = self.regime_detector.assign(emb_t) if self.regime_detector.fitted else None

        # Write to buffer
        self.online_buffer.write(emb_whitened, err_t, regimes=regime_ids)

        # Also seed short buffer if dual buffer is enabled
        if self.use_dual_buffer and self.short_buffer is not None:
            self.short_buffer.write(emb_whitened, err_t, regimes=regime_ids)

        print(f"[BufferSeed] Seeded buffer with {len(emb)} train samples "
              f"(error range: [{err.min():.4f}, {err.max():.4f}], "
              f"non-zero: {(np.abs(err) > 0.01).sum()}/{len(err)})")

        # Free memory
        self._train_embeddings_for_seed = None
        self._train_errors_for_seed = None

    def _adaptive_supported(self) -> bool:
        """Whether the frozen-neighbour gradient covers the configured readout.

        The derivation in the paper differentiates through the distance-weighted
        aggregation of Eqs. (softmax kernel) and (buffer correction). It does not cover
        the regime-aware neighbour penalty or logit-space correction, so the adaptation
        is left untouched when either is active rather than updated by a formula that does
        not correspond to them.
        """
        reason = None
        if getattr(self, "use_regime_knn_penalty", False):
            reason = "the regime-aware neighbour penalty"
        elif getattr(self, "correction_space", "raw") == "logit":
            reason = "logit-space correction"
        elif not self.is_regression:
            reason = "a classification objective (the derivation is in squared-error currency)"

        if reason is None:
            return True
        if self.use_adaptive_embedding and not self._adaptive_unsupported_warned:
            self._adaptive_unsupported_warned = True
            print("[AdaptiveEmbedding] use_adaptive_embedding is set. The "
                  f"frozen-neighbour gradient is derived for {reason}, so U and V keep "
                  "their initial values and the retrieval geometry stays the identity for "
                  "this run. Set use_adaptive_embedding=false to state that in the config.")
        return False

    @staticmethod
    def _weighted_knn(q, keys, values, k, temp, clip):
        """Distance-weighted neighbour aggregation, differentiable in `q`.

        Neighbour indices are chosen without gradient and then held fixed; only the
        distances to those neighbours carry gradient.
        """
        n = keys.shape[0]
        k_act = min(int(k), n)
        if k_act < 1:
            return None
        with torch.no_grad():
            idx = torch.topk(torch.cdist(q, keys), k_act, dim=1, largest=False).indices
        nb = keys[idx]                                   # [B, k, D]
        d = torch.linalg.vector_norm(q.unsqueeze(1) - nb, dim=2)
        v = values[idx]
        logits = -d / max(float(temp), 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        w = torch.exp(logits)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-10)
        if k_act >= 3 and clip is not None:
            wn = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            centre = (wn * v).sum(dim=1, keepdim=True)
            mad = torch.clamp((wn * (v - centre).abs()).sum(dim=1, keepdim=True), min=1e-6)
            v = torch.min(torch.max(v, centre - clip * mad), centre + clip * mad)
        return (w * v).sum(dim=1) / torch.clamp(w.sum(dim=1), min=1e-8)

    def _adaptive_gradient_step(self, raw_emb: torch.Tensor, true_labels: torch.Tensor):
        """One step on U, V using the frozen-neighbour gradient of the batch loss.

        Called before this chunk is written to the buffers, so the neighbour sets are the
        ones that produced the predictions being scored.
        """
        U0, V0 = self._adaptive_U, self._adaptive_V
        if U0 is None or raw_emb.size(0) < 2:
            return
        ctx = getattr(self, "_adaptive_pre_corr", None)
        if ctx is None:
            return
        prior_out, scale = ctx
        if prior_out.shape[0] != raw_emb.shape[0]:
            return
        n_long = self.online_buffer.current_size
        if n_long < self.online_min_samples:
            return

        U = U0.detach().clone().requires_grad_(True)
        V = V0.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            q = raw_emb + (raw_emb @ U) @ V.t()
            lk = self.online_buffer.keys[:n_long]
            lv = self.online_buffer.values[:n_long].squeeze(-1)
            k_long = int(self.dual_buffer_long_k) if self.use_dual_buffer else int(self.online_k)
            corr = self._weighted_knn(q, lk, lv, k_long, self.dual_buffer_long_softmax_temp,
                                      float(self.knn_clip))
            if corr is None:
                return
            if self.use_dual_buffer and self.short_buffer is not None:
                n_short = self.short_buffer.current_size
                k_short = int(self.dual_buffer_short_k)
                if n_short >= min(k_short, 5):
                    sc = self._weighted_knn(q, self.short_buffer.keys[:n_short],
                                            self.short_buffer.values[:n_short].squeeze(-1),
                                            min(k_short, n_short),
                                            self.dual_buffer_short_softmax_temp,
                                            float(self.knn_clip))
                    if sc is not None:
                        w_mix = float(self.dual_buffer_mix_w)
                        corr = w_mix * sc + (1.0 - w_mix) * corr
            pred = prior_out + scale * corr
            loss = ((pred - true_labels.float()) ** 2).mean()
        gU, gV = torch.autograd.grad(loss, [U, V], allow_unused=True)
        if gU is None and gV is None:
            return
        gU = torch.zeros_like(U) if gU is None else gU
        gV = torch.zeros_like(V) if gV is None else gV
        lr, wd = float(self.adaptive_lr), float(self.adaptive_weight_decay)
        with torch.no_grad():
            Un = (U - lr * (gU + wd * U)).detach()
            Vn = (V - lr * (gV + wd * V)).detach()
        if torch.isfinite(Un).all() and torch.isfinite(Vn).all():
            self._adaptive_U, self._adaptive_V = Un, Vn

    def _apply_correction(self, output: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """Apply additive correction, optionally in logit space for classification.

        When correction_space='logit' and output is a probability (1D, classification),
        converts output to logit space, adds correction, converts back. This makes small
        corrections more impactful for AUC since they operate on the log-odds scale.
        """
        if self.correction_space == 'logit' and not self.is_regression and output.dim() == 1:
            # output is P(class=1), convert to logit
            p = output.clamp(1e-6, 1 - 1e-6)
            logit = torch.log(p / (1 - p))
            logit = logit + corr
            output = torch.sigmoid(logit)
        else:
            output = output + corr
        return output

    def set_label_std(self, std: float):
        """Set label std for denormalized gate metric computation."""
        self._label_std = float(std)

    def enable_online_adaptation(self, k: int = 5, min_samples: int = 100, scale: float = 0.5):
        """Enable online test-time adaptation."""
        self.online_mode = True
        self.online_k = k
        self.online_min_samples = min_samples
        self.online_scale = scale
        self.online_buffer.clear()
        if self.use_dual_buffer and self.short_buffer is not None:
            self.short_buffer.clear()
        if self.use_dual_buffer and self.use_regime_short_buffers and self.regime_short_buffers is not None:
            for b in self.regime_short_buffers:
                b.clear()
        self._drift_ema_abs_err = 0.0
        self._drift_err_window = []
        # Reset adaptive embedding
        self._adaptive_U = None
        self._adaptive_V = None
        self._adaptive_embed_dim = None
        self._adaptive_pre_corr = None

        # Init CfC hidden state + online optimizer

        # Seed buffer with training data if enabled
        if self.seed_buffer_from_train and self._train_embeddings_for_seed is not None:
            self._seed_buffer_from_training()

        print(f"Online adaptation enabled: k={k}, min_samples={min_samples}, scale={scale}"
              + (f", buffer_seeded={self.online_buffer.current_size}" if self.seed_buffer_from_train else ""))

    def disable_online_adaptation(self):
        """Disable online adaptation and clear buffer."""
        self.online_mode = False
        self.online_buffer.clear()
        if self.use_dual_buffer and self.short_buffer is not None:
            self.short_buffer.clear()
        if self.use_dual_buffer and self.use_regime_short_buffers and self.regime_short_buffers is not None:
            for b in self.regime_short_buffers:
                b.clear()
        self._drift_ema_abs_err = 0.0
        self._drift_err_window = []
        self._adaptive_U = None
        self._adaptive_V = None
        self._adaptive_embed_dim = None
        self._adaptive_pre_corr = None
        print("Online adaptation disabled")

    def update_online_buffer(self, embeddings: torch.Tensor, predictions: torch.Tensor,
                             true_labels: torch.Tensor):
        """Update online buffer with (embedding, error) pairs after observing ground truth."""
        if not self.online_mode:
            return
        # For classification, predictions may be [batch, n_classes] — reduce to scalar
        if not self.is_regression and predictions.dim() > 1 and predictions.shape[-1] > 1:
            if predictions.shape[-1] == 2:
                predictions = predictions[:, 1]  # P(positive class)
            else:
                predictions = predictions.max(dim=-1).values
        errors = true_labels.float() - predictions
        # drift EMA (Tier2-style drift signal based on observed abs error)
        if self.use_drift_mixw or self.use_drift_k:
            try:
                ae = float(errors.detach().abs().mean().item())
                a = float(self.drift_ema_alpha)
                self._drift_ema_abs_err = (1.0 - a) * float(self._drift_ema_abs_err) + a * ae
            except Exception:
                pass

        # Rolling error window for smooth drift signal
        if self.use_drift_mixw:
            try:
                abs_errs = errors.detach().abs()
                for ae_val in abs_errs.cpu().tolist():
                    self._drift_err_window.append(float(ae_val))
                # Keep window at max size
                while len(self._drift_err_window) > self.drift_window_size:
                    self._drift_err_window.pop(0)
            except Exception:
                pass

        knn_emb_long = self._whiten(embeddings.detach())

        # Apply adaptive embedding projection to buffer keys (same projection as query)
        if self.use_adaptive_embedding and self._adaptive_U is not None:
            D = knn_emb_long.size(1)
            proj = torch.eye(D, device=knn_emb_long.device) + self._adaptive_U @ self._adaptive_V.t()
            knn_emb_long_raw = knn_emb_long.clone()  # pre-projection for SGD
            knn_emb_long = knn_emb_long @ proj

        # Store regime ids if using regime-aware kNN penalty
        regimes = None
        if self.use_regime_knn_penalty:
            try:
                regimes = self.regime_detector.assign(embeddings.detach())
            except Exception:
                regimes = None
        # Gradient step before the buffer write: the neighbour sets used below are the
        # ones that produced the predictions being scored.
        if (self.use_adaptive_embedding and self._adaptive_U is not None
                and self._adaptive_supported()):
            self._adaptive_gradient_step(knn_emb_long_raw, true_labels)

        self.online_buffer.write(knn_emb_long, errors.detach(), regimes=regimes)

        if self.use_dual_buffer and self.short_buffer is not None:
            # Short buffer keys: either raw embeddings or whitened embeddings
            knn_emb_short = embeddings.detach() if self.dual_buffer_short_use_raw else knn_emb_long

            # Optionally filter which samples enter the short-term memory.
            strat = self.short_write_strategy
            if strat == 'all':
                sel = None
            else:
                abs_err = errors.detach().abs().flatten()
                if abs_err.numel() == 0:
                    sel = None
                elif strat == 'topm':
                    m = max(1, min(int(self.short_write_topm), abs_err.numel()))
                    sel = torch.topk(abs_err, k=m, largest=True).indices
                elif strat == 'quantile':
                    q = float(self.short_write_quantile)
                    q = min(1.0, max(0.0, q))
                    thr = torch.quantile(abs_err, q=q)
                    sel = (abs_err >= thr).nonzero(as_tuple=False).flatten()
                elif strat == 'dedup':
                    # De-duplicate near-identical embeddings in the short buffer to improve diversity.
                    # For efficiency, do a per-sample NN distance check against current short keys.
                    eps = float(self.short_write_dedup_eps)
                    if self.short_buffer.current_size <= 0:
                        sel = None
                    else:
                        d = torch.cdist(knn_emb_short, self.short_buffer.keys[: self.short_buffer.current_size])
                        dmin = d.min(dim=-1).values
                        sel = (dmin >= eps).nonzero(as_tuple=False).flatten()
                else:
                    # Unknown strategy: fall back to writing all
                    sel = None

            # Choose which short buffer to write into
            target = self.short_buffer
            if self.use_regime_short_buffers and self.regime_short_buffers is not None:
                # route by regime assignment using raw embeddings
                try:
                    regime_ids = self.regime_detector.assign(embeddings.detach())
                    # if batch contains multiple regimes, write per-regime slices
                    if sel is None:
                        sel_idx = torch.arange(knn_emb_short.size(0), device=knn_emb_short.device)
                    else:
                        sel_idx = sel
                    for r in range(self.regime_detector.n_regimes):
                        m = (regime_ids[sel_idx] == r).nonzero(as_tuple=False).flatten()
                        if m.numel() == 0:
                            continue
                        self.regime_short_buffers[r].write(knn_emb_short[sel_idx[m]], errors.detach()[sel_idx[m]], regimes=None)
                    return
                except Exception:
                    target = self.short_buffer

            if sel is None:
                target.write(knn_emb_short, errors.detach(), regimes=regimes)
            else:
                target.write(knn_emb_short[sel], errors.detach()[sel], regimes=(regimes[sel] if regimes is not None else None))

        # Update drift EMA: track systematic residual bias
        # errors = y_true - slow_pred; after kNN, residual ≈ errors - scale * knn_corr
        # We track mean(errors) as proxy — if kNN consistently under/over-corrects,
        # this captures the remaining bias
        if self.use_drift_correction:
            chunk_mean_err = float(errors.mean().item())
            alpha_d = self.drift_correction_alpha
            if not hasattr(self, '_drift_ema_val'):
                self._drift_ema_val = chunk_mean_err
            else:
                self._drift_ema_val = (1 - alpha_d) * self._drift_ema_val + alpha_d * chunk_mean_err

        # Update CfC temporal stats

        # Update cascaded residual buffer (Layer 2): store (whitened_emb, L1_pred - label)
        if self.use_cascaded_residual and hasattr(self, '_output_before_l2') and self._output_before_l2 is not None:
            with torch.no_grad():
                if self.cascaded_use_dual_buffer:
                    # Dual-buffer mode: short + long buffers
                    if self._cascaded_short_buffer is None:
                        self._cascaded_short_buffer = OnlineKNNBuffer(
                            max_size=self.cascaded_short_buf2,
                            device=str(embeddings.device),
                            temporal_decay=1.0,
                            knn_kernel='softmax',
                            knn_agg='clip',
                            knn_trim_q=0.1,
                            knn_clip=self.cascaded_clip,
                            knn_softmax_temp=self.cascaded_short_temp2,
                        )
                    if self._cascaded_long_buffer is None:
                        self._cascaded_long_buffer = OnlineKNNBuffer(
                            max_size=self.cascaded_long_buf2,
                            device=str(embeddings.device),
                            temporal_decay=1.0,
                            knn_kernel='softmax',
                            knn_agg='clip',
                            knn_trim_q=0.1,
                            knn_clip=self.cascaded_clip,
                            knn_softmax_temp=self.cascaded_long_temp2,
                        )
                    l2_keys = self._whiten(embeddings.detach())
                    l2_residuals = self._output_before_l2[:embeddings.size(0)] - true_labels
                    self._cascaded_short_buffer.write(l2_keys, l2_residuals.detach())
                    self._cascaded_long_buffer.write(l2_keys, l2_residuals.detach())
                    self._cascaded_samples_seen += embeddings.size(0)
                else:
                    # Single-buffer mode (original)
                    if self._cascaded_buffer is None:
                        self._cascaded_buffer = OnlineKNNBuffer(
                            max_size=self.cascaded_buf_size,
                            device=str(embeddings.device),
                            temporal_decay=1.0,
                            knn_kernel='softmax',
                            knn_agg='clip',
                            knn_trim_q=0.1,
                            knn_clip=self.cascaded_clip,
                            knn_softmax_temp=self.cascaded_temp,
                        )
                    l2_keys = self._whiten(embeddings.detach())
                    # Residuals: prediction - label (positive = overshoot)
                    l2_residuals = self._output_before_l2[:embeddings.size(0)] - true_labels
                    self._cascaded_buffer.write(l2_keys, l2_residuals.detach())
                    self._cascaded_samples_seen += embeddings.size(0)

                # Adaptive gate: compare error with vs without L2
                if (self._cascaded_adaptive and not self._cascaded_disabled
                        and hasattr(self, '_output_after_l2') and self._output_after_l2 is not None):
                    B = min(embeddings.size(0), self._output_before_l2.size(0), self._output_after_l2.size(0))
                    # Compute benefit in denormalized space (multiply by label_std²)
                    std2 = self._label_std ** 2
                    err_without = (self._output_before_l2[:B] - true_labels[:B]).pow(2).mean().item() * std2
                    err_with = (self._output_after_l2[:B] - true_labels[:B]).pow(2).mean().item() * std2
                    benefit = err_without - err_with  # positive = L2 helps (denormalized MSE)
                    self._cascaded_benefit_history.append(benefit)
                    alpha = self._cascaded_ema_alpha
                    self._cascaded_ema_benefit = (1 - alpha) * self._cascaded_ema_benefit + alpha * benefit

                    # Diagnostic logging (every 1000 samples)
                    if self._cascaded_l2_applied_count > 0 and self._cascaded_l2_applied_count % 1000 < B:
                        recent = self._cascaded_benefit_history[-100:]
                        frac_hurt = sum(1 for b in recent if b < 0) / max(len(recent), 1)
                        print(f"[L2 Gate] samples={self._cascaded_l2_applied_count}, "
                              f"EMA={self._cascaded_ema_benefit:.8f} (denorm, std={self._label_std:.4f}), "
                              f"hurt_frac={frac_hurt:.2f}, "
                              f"batch_benefit={benefit:.8f}")

                    # Only evaluate after enough L2-corrected samples for stable signal
                    if self._cascaded_l2_applied_count > self._cascaded_eval_after:
                        # Use both criteria: EMA negative AND majority of recent batches hurt
                        recent = self._cascaded_benefit_history[-200:]  # last 200 batches
                        frac_hurt = sum(1 for b in recent if b < 0) / max(len(recent), 1)
                        # Threshold scales with label variance (denormalized space)
                        threshold = self._cascaded_disable_threshold * (self._label_std ** 2)
                        if self._cascaded_ema_benefit < threshold and frac_hurt > 0.6:
                            # L2 consistently hurting (>60% of recent batches negative + negative EMA)
                            self._cascaded_disabled = True
                            print(f"[Cascaded L2] Adaptive gate DISABLED L2 after {self._cascaded_l2_applied_count} samples "
                                  f"(EMA={self._cascaded_ema_benefit:.6f}, hurt_frac={frac_hurt:.2f})")


    def get_online_stats(self) -> Dict[str, Any]:
        """Get online adaptation statistics."""
        stats = {
            'online_mode': self.online_mode,
            'online_buffer_size': self.online_buffer.current_size,
            'online_k': self.online_k,
            'online_min_samples': self.online_min_samples,
            'online_scale': self.online_scale,
            'online_scale_mult': float(self.online_scale_mult),
            'embedding_quality_mult': float(self._embedding_quality_mult),
            'fit_global_online_scale': bool(self.fit_global_online_scale),
            'use_regime_online_scale': self.use_regime_online_scale,
        }
        if self.use_regime_online_scale and self.regime_detector.online_scale_by_regime is not None:
            stats['regime_online_scales'] = self.regime_detector.online_scale_by_regime.cpu().tolist()
        return stats

    # =========================================================================
    # Forward pass
    # =========================================================================

    def forward(
        self,
        x: torch.Tensor,
        x_cat: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        split: str = 'train',
    ) -> torch.Tensor:
        """
        Three-tier forward pass.

        output = slow_pred + regime_scale * online_scale * knn_correction + regime_bias

        Args:
            x: [batch, d_in] — input features
            x_cat: ignored (compatibility)
            timestamps: ignored (compatibility)
            labels: [batch] — ground truth (used during bio-learning for buffer writes)
            indices: [batch] — sample indices for cache lookup
            split: 'train', 'val', or 'test'

        Returns:
            output: [batch] — predictions
        """
        batch_size = x.size(0)
        device = x.device

        # =====================================================================
        # Tier 1: Slow Prior — base prediction + raw embeddings
        # =====================================================================
        slow_result = self.slow_prior(x, indices=indices, split=split)
        slow_pred = slow_result.get('prediction', None)
        raw_embeddings = slow_result.get('raw_embeddings', None)

        # Fall back to projected features if raw embeddings not available
        if raw_embeddings is None:
            raw_embeddings = slow_result['features']

        # Ensure correct shape
        if slow_pred is not None:
            if slow_pred.dim() > 1:
                slow_pred = slow_pred.squeeze(-1)
            slow_pred = slow_pred.detach()

        # Tree leaf embeddings override for kNN (classification only)
        knn_embeddings = raw_embeddings
        if indices is not None:
            tree_leaf_emb = self._get_tree_leaf_emb(indices, split)
            if tree_leaf_emb is not None:
                knn_embeddings = tree_leaf_emb

        # Store for external access (used by method's predict() for buffer updates)
        self._teacher_pred = slow_pred
        self._last_raw_embeddings = knn_embeddings.detach()

        # Start with slow prior prediction
        output = slow_pred if slow_pred is not None else torch.zeros(batch_size, device=device)

        if not self.is_regression and output.dim() == 3:
             # Collapse the extra dimension a multi-class head can produce
             if output.size(1) == 1:
                 output = output.squeeze(1)
             else:
                 output = output.mean(dim=1)

        # Optional: Ensemble prior integration (Tier 2)
        if (self.use_ensemble_gating and indices is not None
                and self._ensemble_prior_names):
            ensemble_preds = self._get_ensemble_predictions(indices, split)
            if ensemble_preds is not None:
                avail = self._ensemble_split_available.get(split)
                if avail is not None:
                    avail = avail.to(device)
                if self.ensemble_mode == 'adaptive' and self._ensemble_gating is not None:
                    # Adaptive: per-sample weights from gating MLP
                    emb_w = self._whiten(raw_embeddings.detach())
                    with torch.no_grad():
                        gating_weights = self._ensemble_gating(emb_w)  # [B, n_priors]
                    gating_weights = self._renormalise_over_available(gating_weights, avail)
                    output = (gating_weights * ensemble_preds).sum(dim=1)
                elif self.ensemble_mode in ('fixed', 'online') and self._ensemble_fixed_weights is not None:
                    # Fixed/Online: global weights (online mode updates weights externally)
                    w = self._ensemble_fixed_weights.to(device)  # [n_priors]
                    w = self._renormalise_over_available(w.unsqueeze(0), avail).squeeze(0)
                    output = (w.unsqueeze(0) * ensemble_preds).sum(dim=1)
                else:
                    # Fallback: equal weights over the priors present for this split
                    if avail is not None and not bool(avail.all()):
                        m = avail.to(ensemble_preds.dtype)
                        output = (ensemble_preds * m).sum(dim=1) / m.sum().clamp(min=1.0)
                    else:
                        output = ensemble_preds.mean(dim=1)
                # Update teacher_pred to reflect ensemble blend
                self._teacher_pred = output.detach()
                # Store for Scheme C disagreement scaling
                self._last_ensemble_preds = ensemble_preds.detach()
            else:
                self._last_ensemble_preds = None
        else:
            self._last_ensemble_preds = None

        # Optional: TabPFN auxiliary gating signal (does not replace slow_pred)
        if self.use_tabpfn_aux and self.tabpfn_aux is not None and indices is not None:
            try:
                aux = self.tabpfn_aux(x, indices=indices, split=split)
                tabpfn_pred = aux.get('prediction', None)
                if tabpfn_pred is not None:
                    if tabpfn_pred.dim() > 1:
                        tabpfn_pred = tabpfn_pred.squeeze(-1)
                    delta = (tabpfn_pred.detach() - output).clamp(min=-self.tabpfn_aux_clip, max=self.tabpfn_aux_clip)
                    output = output + float(self.tabpfn_aux_weight) * delta
            except Exception:
                pass

        # =====================================================================
        # Tier 2: Regime Modulation — per-regime bias + scale
        # =====================================================================
        regime_ids = self.regime_detector.assign(knn_embeddings.detach())
        regime_scales, regime_biases = self.regime_detector.get_params(regime_ids)

        # Optional: add regime bias (systematic error correction per region)
        if self.use_regime_bias:
            output = output + regime_biases

        # =====================================================================
        # Tier 3: Online k-NN Correction — euclidean distance based
        # =====================================================================
        if self.online_mode and self.online_buffer.current_size >= self.online_min_samples:
            knn_emb_long = self._whiten(knn_embeddings.detach())

            # Online Embedding Adaptation: apply low-rank projection
            if self.use_adaptive_embedding:
                D = knn_emb_long.size(1)
                if self._adaptive_U is None:
                    self._adaptive_embed_dim = D
                    # Asymmetric initialisation, as in LoRA: the down-projection is drawn
                    # at random and the up-projection is zero, so that U V^T = 0 at t = 0
                    # while the gradient is not. Initialising BOTH factors to zero makes
                    # (0, 0) an exact fixed point of the update below -- grad_U carries a
                    # factor of V and grad_V a factor of U -- and the adaptation can never
                    # leave its initial value.
                    sigma0 = 1.0 / math.sqrt(3.0 * D)
                    self._adaptive_U = torch.randn(D, self.adaptive_rank, device=device) * sigma0
                    self._adaptive_V = torch.zeros(D, self.adaptive_rank, device=device)
                # e_adapted = e_whitened @ (I + U @ V^T)
                proj = torch.eye(D, device=device) + self._adaptive_U @ self._adaptive_V.t()
                knn_emb_long = knn_emb_long @ proj
                self._last_knn_emb_raw = self._whiten(knn_embeddings.detach())  # store pre-projection for SGD

            knn_emb_short = knn_embeddings.detach() if self.dual_buffer_short_use_raw else knn_emb_long

            # Optionally schedule k based on drift
            k_long = int(self.dual_buffer_long_k) if self.use_dual_buffer else int(self.online_k)
            k_short = int(self.dual_buffer_short_k) if self.use_dual_buffer else int(self.online_k)
            if self.use_drift_k:
                try:
                    kk = int(self.drift_k_high) if float(self._drift_ema_abs_err) > float(self.drift_threshold) else int(self.drift_k_low)
                    k_long = kk
                    k_short = kk
                except Exception:
                    pass

            # Optionally use different softmax temps for long/short buffers
            try:
                base_T_long = float(self.dual_buffer_long_softmax_temp)
                if self.use_regime_conf_temp and self.regime_detector.fitted:
                    # confidence: distance to centroid (raw embedding space)
                    cent = self.regime_detector.centroids[regime_ids]  # [B,D]
                    dist_c = (knn_embeddings.detach() - cent).pow(2).sum(dim=-1).sqrt()  # [B]
                    med = float(self.regime_detector.centroid_dist_median)
                    med = max(1e-6, med)
                    # Only increase temperature when farther than median for beta>=0.
                    # For beta<0, allow symmetric response (can get sharper when far).
                    beta = float(self.regime_conf_temp_beta_long)
                    if beta >= 0:
                        excess = (dist_c / med - 1.0).clamp(min=0.0)
                    else:
                        excess = (dist_c / med - 1.0)
                    # use mean excess as a simple scalar modulator
                    T_eff = base_T_long * (1.0 + beta * float(excess.mean().item()))
                    self.online_buffer.knn_softmax_temp = float(max(1e-6, T_eff))
                else:
                    self.online_buffer.knn_softmax_temp = base_T_long
            except Exception:
                pass

            # regime-aware penalty (Tier2 prior)
            qreg = regime_ids if self.use_regime_knn_penalty else None
            lam_long = float(self.regime_knn_lambda_long) if self.use_regime_knn_penalty else 0.0
            clip_long = None
            if self.use_regime_clip:
                try:
                    clip_long = (float(self.knn_clip) * regime_scales).clamp(min=float(self.regime_clip_min), max=float(self.regime_clip_max))
                except Exception:
                    clip_long = None

            # Use Hopfield, cross-attention, or standard kNN
            knn_correction = self.online_buffer.query(
                knn_emb_long,
                k=k_long,
                query_regimes=qreg,
                regime_mismatch_lambda=lam_long,
                clip_override=clip_long,
            )
            if self.use_dual_buffer and self.short_buffer is not None:
                # Mix weight (optionally drift-scheduled)
                w = float(self.dual_buffer_mix_w)
                if self.use_drift_mixw:
                    try:
                        wnd = self._drift_err_window
                        if len(wnd) >= self.drift_window_size:
                            # Smooth sigmoid drift mechanism
                            half = self.drift_window_size // 2
                            older = sum(wnd[:half]) / half
                            recent = sum(wnd[half:]) / half
                            drift_signal = recent / max(older, 1e-8)
                            x_sig = self.drift_alpha * (drift_signal - 1.0)
                            x_sig = max(-20.0, min(20.0, x_sig))
                            sigmoid_val = 1.0 / (1.0 + math.exp(-x_sig))
                            w = self.drift_base_mix_w * sigmoid_val
                        else:
                            # Before enough data: use half of base_mix_w
                            w = self.drift_base_mix_w * 0.5
                    except Exception:
                        w = float(self.dual_buffer_mix_w)

                if self.use_regime_short_buffers and self.regime_short_buffers is not None:
                    # Query regime buffers.
                    # - hard: query assigned regime only
                    # - soft2: query top-2 regimes by centroid distance and mix
                    knn_short = torch.zeros_like(knn_correction)
                    if self.regime_short_mode == 'soft2' and self.regime_detector.fitted and self.regime_detector.centroids is not None:
                        dists = torch.cdist(knn_embeddings.detach(), self.regime_detector.centroids)  # [B, R]
                        top2_d, top2_idx = torch.topk(dists, k=2, largest=False)
                        tau = max(1e-6, float(self.regime_short_tau))
                        w2 = torch.softmax(-top2_d / tau, dim=-1)  # [B,2]
                        # Accumulate contributions from the two regimes
                        for j in range(2):
                            r_ids = top2_idx[:, j]
                            for r in range(self.regime_detector.n_regimes):
                                m = (r_ids == r).nonzero(as_tuple=False).flatten()
                                if m.numel() == 0:
                                    continue
                                buf = self.regime_short_buffers[r]
                                if buf.current_size < 1:
                                    continue
                                try:
                                    buf.knn_softmax_temp = float(self.dual_buffer_short_softmax_temp)
                                except Exception:
                                    pass
                                q = buf.query(knn_emb_short[m], k=k_short)
                                wt = w2[m, j]
                                # Ensure shape compatibility for both [n] and [n,1]
                                if q.dim() > wt.dim():
                                    wt = wt.unsqueeze(-1)
                                knn_short[m] += wt * q
                    else:
                        for r in range(self.regime_detector.n_regimes):
                            m = (regime_ids == r).nonzero(as_tuple=False).flatten()
                            if m.numel() == 0:
                                continue
                            buf = self.regime_short_buffers[r]
                            if buf.current_size < 1:
                                continue
                            lam_short = float(self.regime_knn_lambda_short) if self.use_regime_knn_penalty else 0.0
                            clip_short_m = None
                            if self.use_regime_clip:
                                try:
                                    clip_short_m = (float(self.knn_clip) * regime_scales[m]).clamp(min=float(self.regime_clip_min), max=float(self.regime_clip_max))
                                except Exception:
                                    clip_short_m = None
                            knn_short[m] = buf.query(
                                knn_emb_short[m],
                                k=k_short,
                                query_regimes=qreg[m],
                                regime_mismatch_lambda=lam_short,
                                clip_override=clip_short_m,
                            )
                else:
                    if self.short_buffer.current_size < 1:
                        knn_short = None
                    else:
                        try:
                            base_T_short = float(self.dual_buffer_short_softmax_temp)
                            if self.use_regime_conf_temp and self.regime_detector.fitted:
                                cent = self.regime_detector.centroids[regime_ids]
                                dist_c = (knn_embeddings.detach() - cent).pow(2).sum(dim=-1).sqrt()
                                med = float(self.regime_detector.centroid_dist_median)
                                med = max(1e-6, med)
                                beta = float(self.regime_conf_temp_beta_short)
                                if beta >= 0:
                                    excess = (dist_c / med - 1.0).clamp(min=0.0)
                                else:
                                    excess = (dist_c / med - 1.0)
                                T_eff = base_T_short * (1.0 + beta * float(excess.mean().item()))
                                self.short_buffer.knn_softmax_temp = float(max(1e-6, T_eff))
                            else:
                                self.short_buffer.knn_softmax_temp = base_T_short
                        except Exception:
                            pass
                        lam_short = float(self.regime_knn_lambda_short) if self.use_regime_knn_penalty else 0.0
                        clip_short = None
                        if self.use_regime_clip:
                            try:
                                clip_short = (float(self.knn_clip) * regime_scales).clamp(min=float(self.regime_clip_min), max=float(self.regime_clip_max))
                            except Exception:
                                clip_short = None
                        knn_short = self.short_buffer.query(
                            knn_emb_short,
                            k=k_short,
                            query_regimes=qreg,
                            regime_mismatch_lambda=lam_short,
                            clip_override=clip_short,
                        )

                if knn_short is not None:
                    # Optional learned per-sample gate for mixing
                    if self.use_mixw_gate and self.mixw_gate is not None:
                        try:
                            # regime confidence feature (dist to centroid / median)
                            if self.regime_detector.fitted and self.regime_detector.centroids is not None:
                                cent = self.regime_detector.centroids[regime_ids]
                                dist_c = (knn_embeddings.detach() - cent).pow(2).sum(dim=-1).sqrt()
                                med = max(1e-6, float(self.regime_detector.centroid_dist_median))
                                dist_norm = (dist_c / med).clamp(min=0.0)
                            else:
                                dist_norm = torch.zeros_like(knn_correction)

                            long_corr = knn_correction.detach()
                            short_corr = knn_short.detach()
                            diff = (short_corr - long_corr).detach()
                            feats = torch.stack([
                                dist_norm,
                                long_corr.abs(),
                                short_corr.abs(),
                                diff.abs(),
                                torch.ones_like(dist_norm),
                            ], dim=-1)  # [B,5]
                            w = self.mixw_gate(feats).squeeze(-1)  # [B]
                        except Exception:
                            w = float(w)
                    knn_correction = w * knn_short + (1.0 - w) * knn_correction

            # Compute effective scale
            effective_scale = self.online_scale * float(self.online_scale_mult) * float(self._embedding_quality_mult)
            if self.use_regime_online_scale:
                regime_online_scales = self.regime_detector.get_online_scales(regime_ids)
                effective_scale = effective_scale * regime_online_scales

            # Adaptive scale: modulate per-sample based on kNN confidence
            if self.use_adaptive_scale:
                with torch.no_grad():
                    n_mem = self.online_buffer.current_size
                    k_q = min(int(self.online_k), n_mem)
                    dists = torch.cdist(knn_emb_long, self.online_buffer.keys[:n_mem])
                    topk_dists, _ = torch.topk(dists, k_q, largest=False, dim=-1)  # [B, k]
                    topk_idx = torch.topk(dists, k_q, largest=False, dim=-1)[1]
                    topk_vals = self.online_buffer.values[:n_mem][topk_idx].squeeze(-1)  # [B, k]

                    # Confidence from distance: closer = higher confidence
                    mean_dist = topk_dists.mean(dim=-1)  # [B]
                    # Use running stats for normalization
                    if not hasattr(self, '_dist_ema'):
                        self._dist_ema = float(mean_dist.mean().item())
                        self._dist_ema_sq = float((mean_dist ** 2).mean().item())
                    else:
                        alpha = self.adaptive_scale_ema_alpha
                        md = float(mean_dist.mean().item())
                        self._dist_ema = (1 - alpha) * self._dist_ema + alpha * md
                        self._dist_ema_sq = (1 - alpha) * self._dist_ema_sq + alpha * md ** 2

                    dist_std = max(1e-6, (self._dist_ema_sq - self._dist_ema ** 2) ** 0.5)
                    # z-score: how far this batch is relative to typical
                    z_dist = (mean_dist - self._dist_ema) / dist_std  # [B], positive = farther than usual

                    # Agreement: lower std of neighbor values = more agreement
                    val_std = topk_vals.std(dim=-1)  # [B]
                    if not hasattr(self, '_valstd_ema'):
                        self._valstd_ema = float(val_std.mean().item())
                    else:
                        self._valstd_ema = (1 - alpha) * self._valstd_ema + alpha * float(val_std.mean().item())

                    z_agree = val_std / max(1e-6, self._valstd_ema)  # [B], >1 = more disagreement

                    # Confidence factor: reduce scale when far or disagreeing
                    # conf in [adaptive_scale_min, adaptive_scale_max]
                    conf = 1.0 - self.adaptive_scale_dist_w * z_dist.clamp(min=-2, max=2) \
                           - self.adaptive_scale_agree_w * (z_agree - 1.0).clamp(min=-1, max=2)
                    conf = conf.clamp(min=self.adaptive_scale_min, max=self.adaptive_scale_max)

                    effective_scale = effective_scale * conf

            # Scheme C: Disagreement-driven correction scaling
            # When ensemble priors disagree, boost kNN correction strength
            if self.use_disagree_scale and self._last_ensemble_preds is not None:
                with torch.no_grad():
                    # Compute disagreement as weighted deviation from ensemble mean
                    # Each prior's contribution is weighted, so measure spread of
                    # weighted predictions around the blend (like weighted variance)
                    ep = self._last_ensemble_preds  # [B, n_priors]
                    blend = output.unsqueeze(1)  # [B, 1] — the weighted mean
                    if self._ensemble_fixed_weights is not None:
                        w = self._ensemble_fixed_weights.to(ep.device).unsqueeze(0)  # [1, n_priors]
                        # Weighted std: sqrt(sum(w_i * (p_i - blend)^2))
                        disagree = (w * (ep - blend).pow(2)).sum(dim=1).sqrt()  # [B]
                    else:
                        disagree = (ep - blend).pow(2).mean(dim=1).sqrt()  # [B]
                    d_mean = float(disagree.mean().item())
                    d_std = float(disagree.std().item())
                    alpha = self.disagree_ema_alpha
                    if self._disagree_ema_mean is None:
                        self._disagree_ema_mean = d_mean
                        self._disagree_ema_std = max(d_std, 1e-6)
                    else:
                        self._disagree_ema_mean = (1 - alpha) * self._disagree_ema_mean + alpha * d_mean
                        self._disagree_ema_std = (1 - alpha) * self._disagree_ema_std + alpha * max(d_std, 1e-6)
                    # z-score: positive = more disagreement than usual
                    z_disagree = (disagree - self._disagree_ema_mean) / max(self._disagree_ema_std, 1e-6)
                    # Scale multiplier: 1 + strength * z, clamped
                    disagree_mult = (1.0 + self.disagree_strength * z_disagree).clamp(
                        min=self.disagree_scale_min, max=self.disagree_scale_max
                    )
                    effective_scale = effective_scale * disagree_mult

            # Meta-correction MLP: replace fixed scaling with learned correction
            if self.use_regime_bias:
                corr = regime_scales * effective_scale * knn_correction
                self._adaptive_pre_corr = (output.detach().clone(),
                                           (regime_scales * effective_scale).detach()
                                           if torch.is_tensor(regime_scales) else effective_scale)
                if not self.is_regression and output.dim() == 2 and corr.dim() == 1:
                    corr = corr.unsqueeze(-1)  # [batch, 1] broadcasts to all classes
                output = self._apply_correction(output, corr)
            else:
                corr = effective_scale * knn_correction
                self._adaptive_pre_corr = (output.detach().clone(), effective_scale)
                if not self.is_regression and output.dim() == 2 and corr.dim() == 1:
                    corr = corr.unsqueeze(-1)  # [batch, 1] broadcasts to all classes
                output = self._apply_correction(output, corr)

        # =====================================================================
        # Drift EMA correction: track and correct systematic bias
        # =====================================================================
        if self.online_mode and self.use_drift_correction:
            with torch.no_grad():
                if not hasattr(self, '_drift_ema_val'):
                    self._drift_ema_val = 0.0
                output = output + self._drift_ema_val

        # =====================================================================
        # Tier 3.5: CfC Temporal Correction — additive temporal signal
        # =====================================================================

        # =====================================================================
        # Layer 2: Cascaded Residual Correction — second kNN pass on residuals
        # =====================================================================
        # Store pre-L2 output for residual buffer writes
        if self.use_cascaded_residual:
            self._output_before_l2 = output.detach().clone()

        self._output_after_l2 = None  # reset
        if (self.online_mode and self.use_cascaded_residual
                and not self._cascaded_disabled
                and self._cascaded_samples_seen >= self.cascaded_warmup):
            if self.cascaded_use_dual_buffer:
                # Dual-buffer L2: blend short and long buffer corrections
                lb = self._cascaded_long_buffer
                sb = self._cascaded_short_buffer
                if (lb is not None and sb is not None
                        and lb.current_size >= min(self.cascaded_k, self.cascaded_warmup)):
                    with torch.no_grad():
                        knn_emb_l2 = self._whiten(knn_embeddings.detach())
                        k_long = min(self.cascaded_k, lb.current_size)
                        long_corr = lb.query(knn_emb_l2, k=k_long)
                        # Short buffer: only use if has enough samples
                        if sb.current_size >= min(self.cascaded_k, 5):
                            k_short = min(self.cascaded_k, sb.current_size)
                            short_corr = sb.query(knn_emb_l2, k=k_short)
                            l2_correction = (self.cascaded_dual_mix2 * short_corr
                                             + (1.0 - self.cascaded_dual_mix2) * long_corr)
                        else:
                            l2_correction = long_corr
                        output = output - self.cascaded_scale * l2_correction
                        self._cascaded_l2_applied_count += batch_size
                        self._output_after_l2 = output.detach().clone()
            elif self._cascaded_buffer is not None:
                # Single-buffer L2 (original)
                with torch.no_grad():
                    cb = self._cascaded_buffer
                    if cb.current_size >= min(self.cascaded_k, self.cascaded_warmup):
                        knn_emb_l2 = self._whiten(knn_embeddings.detach())
                        old_temp = cb.knn_softmax_temp
                        old_clip = cb.knn_clip
                        cb.knn_softmax_temp = self.cascaded_temp
                        cb.knn_clip = self.cascaded_clip
                        l2_correction = cb.query(knn_emb_l2, k=min(self.cascaded_k, cb.current_size))
                        cb.knn_softmax_temp = old_temp
                        cb.knn_clip = old_clip
                        output = output - self.cascaded_scale * l2_correction
                        self._cascaded_l2_applied_count += batch_size
                        self._output_after_l2 = output.detach().clone()

        self.step_count += 1

        # For classification: convert scalar prediction to [B, n_classes] logits
        # Only needed when output is 1D (scalar probability), not when already [B, n_classes]
        if self.d_out > 1 and output.dim() == 1:
            # output is P(class=1) from LimiX, clamp to valid range
            p = output.clamp(1e-6, 1 - 1e-6)
            # Convert to logits: log(p / (1-p)) for class 1, 0 for class 0
            logit = torch.log(p / (1 - p))
            output = torch.stack([torch.zeros_like(logit), logit], dim=-1)  # [B, 2]

        if not self.is_regression and output.dim() == 3:
            if output.size(1) == 1:
                output = output.squeeze(1)
            else:
                output = output.mean(dim=1)

        return output

    # =========================================================================
    # Utility methods
    # =========================================================================

    def update_ensemble_weights_online(self, y_true: torch.Tensor, ema_alpha: float = 0.05):
        """Update ensemble weights based on observed prediction errors (online mode).
        
        Tracks per-prior EMA of absolute errors. Re-weights inversely proportional
        to recent error (better priors get higher weight).
        
        Args:
            y_true: [B] ground truth labels for the last forward pass
            ema_alpha: smoothing factor for error EMA
        """
        if (self.ensemble_mode != 'online'
                or self._last_ensemble_preds is None
                or self._ensemble_fixed_weights is None):
            return

        with torch.no_grad():
            ep = self._last_ensemble_preds  # [B, n_priors]
            y = y_true.unsqueeze(1)  # [B, 1]
            # Per-prior absolute error for this chunk
            chunk_errors = (ep - y).abs().mean(dim=0)  # [n_priors]

            # Initialize or update EMA
            if not hasattr(self, '_ensemble_error_ema'):
                self._ensemble_error_ema = chunk_errors.cpu()
            else:
                self._ensemble_error_ema = (
                    (1 - ema_alpha) * self._ensemble_error_ema
                    + ema_alpha * chunk_errors.cpu()
                )

            # Convert errors to weights: inverse error, softmax normalized
            inv_err = 1.0 / (self._ensemble_error_ema + 1e-6)
            new_weights = inv_err / inv_err.sum()

            # Blend: use online_ensemble_blend (default 0.5 = 50/50)
            blend_rate = getattr(self, 'online_ensemble_blend', 0.5)
            orig = self._ensemble_fixed_weights.cpu()
            blended = (1 - blend_rate) * orig + blend_rate * new_weights
            blended = blended / blended.sum()  # renormalize

            self._ensemble_fixed_weights = blended

            # Debug: print weight evolution periodically
            if not hasattr(self, '_online_weight_updates'):
                self._online_weight_updates = 0
            self._online_weight_updates += 1
            if self._online_weight_updates % 100 == 0:
                w_str = ', '.join(f'{n}:{w:.4f}' for n, w in zip(self._ensemble_prior_names, blended.tolist()))
                print(f"  [Online weights @{self._online_weight_updates}] {w_str}")

    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        stats = {
            'step_count': self.step_count,
            'slow_prior_type': self.slow_prior_type,
            'regime_fitted': self.regime_detector.fitted,
            'buffer_size': self.online_buffer.current_size,
        }
        if self.regime_detector.fitted:
            stats['regime_biases'] = self.regime_detector.biases.cpu().tolist()
            stats['regime_scales'] = self.regime_detector.scales.cpu().tolist()
        if self.use_regime_online_scale and self.regime_detector.online_scale_by_regime is not None:
            stats['regime_online_scales'] = self.regime_detector.online_scale_by_regime.cpu().tolist()
        return stats

    def get_correction_only_parameters(self, base_lr: float = 1e-3) -> List[Dict]:
        """No trainable correction parameters in bio mode — return empty."""
        return []
