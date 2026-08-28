"""
Biologically-Inspired Method for Three-Tier PFC Architecture.

Training:
1. Single pass through training data to accumulate embeddings + errors
2. Fit k-means regime detector on accumulated statistics
3. No gradient-based training (bio-compatible)

Inference:
1. Slow prior (LimiX) provides base predictions + raw embeddings
2. Regime detector adds per-regime bias
3. Online k-NN buffer provides error corrections from test-time feedback
4. Mini-batch processing (chunk=64) so earlier samples help later ones
"""

import time
import os
import os.path as osp
import json
import torch
import numpy as np
from typing import Optional, Dict, Any
from torch.cuda.amp import autocast as cuda_autocast

from model.methods.base_temporal import Method_Temporal
from model.models.three_tier_pfc_bio import ThreeTierPFC
from model.utils import Averager


def _with_context_size(slow_prior_config, context_size):
    """Let a configuration set the in-context sample count at the model level.

    Every shipped configuration carries `context_size` next to `slow_prior_config`, which
    holds the same number. Folding the outer one in keeps the two from drifting apart and
    stops the more visible of the two from being a key that edits do nothing to.
    """
    cfg = dict(slow_prior_config or {})
    if context_size is not None:
        if cfg.get("context_size") not in (None, context_size):
            print(f"  config sets context_size {context_size} at the model level and "
                  f"{cfg['context_size']} inside slow_prior_config; using {context_size}")
        cfg["context_size"] = context_size
    return cfg


class ThreeTierPFCBioMethod(Method_Temporal):
    """
    Biologically-inspired training method for Three-Tier PFC Architecture.

    Training: single pass → fit regime detector (no gradients).
    Inference: slow_pred + regime_bias + regime_scale * online_scale * knn_correction.
    """

    def __init__(self, args, is_regression):
        super().__init__(args, is_regression)

    def construct_model(self, model_config=None):
        """Construct the Three-Tier PFC model."""
        if model_config is None:
            model_config = self.args.config['model'].copy()
        else:
            model_config = model_config.copy()

        # Extract method-level params (not passed to model)
        self.finetune_epochs = model_config.pop('finetune_epochs', 0)
        self.use_online_adaptation = model_config.pop('use_online_adaptation', True)

        # Keys the model constructor does not take
        for key in ['n_memory_passes', 'use_tta', 'tta_steps', 'tta_lr',
                     'warmup_epochs', 'freeze_after_warmup']:
            model_config.pop(key, None)

        # Build model config
        config = {
            'd_in': self.d_in,
            'd_out': self.d_out,
            'is_regression': self.is_regression,
            'slow_prior_type': model_config.pop('slow_prior_type', 'limix'),
            'slow_prior_config': _with_context_size(
                model_config.pop('slow_prior_config', {'context_size': 10000}),
                model_config.pop('context_size', None)),
            'regime_k': model_config.pop('regime_k', model_config.pop('n_regimes', 8)),
            'regime_fit_subsample': model_config.pop('regime_fit_subsample', None),
            'online_buffer_size': model_config.pop('online_buffer_size', model_config.pop('buffer_size', 2048)),
            'online_k': model_config.pop('online_k', 5),
            'online_scale': model_config.pop('online_scale', 0.5),
            'online_min_samples': model_config.pop('online_min_samples', 100),
            'online_chunk_size': model_config.pop('online_chunk_size', 64),
            'temporal_decay': model_config.pop('temporal_decay', 1.0),
            # Robust kNN readout options (default preserves existing behavior)
            'knn_kernel': model_config.pop('knn_kernel', 'inverse'),
            'knn_agg': model_config.pop('knn_agg', 'mean'),
            'knn_trim_q': model_config.pop('knn_trim_q', 0.1),
            'knn_clip': model_config.pop('knn_clip', 3.0),
            'knn_softmax_temp': model_config.pop('knn_softmax_temp', 2.0),
            # Global online_scale calibration (val-fitted scalar)
            'fit_global_online_scale': model_config.pop('fit_global_online_scale', False),
            'global_online_scale_ridge': model_config.pop('global_online_scale_ridge', 1e-3),
            'global_online_scale_clip': model_config.pop('global_online_scale_clip', 2.0),
            # Dual-timescale buffer
            'use_dual_buffer': model_config.pop('use_dual_buffer', False),
            'dual_buffer_size': model_config.pop('dual_buffer_size', 512),
            'dual_buffer_mix_w': model_config.pop('dual_buffer_mix_w', 0.2),
            # Learned gate for mix_w
            'use_mixw_gate': model_config.pop('use_mixw_gate', False),
            'mixw_gate_path': model_config.pop('mixw_gate_path', None),
            'mixw_gate_feat_dim': model_config.pop('mixw_gate_feat_dim', 5),
            'dual_buffer_long_decay': model_config.pop('dual_buffer_long_decay', model_config.get('temporal_decay', 1.0)),
            'dual_buffer_short_decay': model_config.pop('dual_buffer_short_decay', model_config.get('temporal_decay', 1.0)),
            'dual_buffer_short_k': model_config.pop('dual_buffer_short_k', model_config.get('online_k', 20)),
            'dual_buffer_long_k': model_config.pop('dual_buffer_long_k', model_config.get('online_k', 20)),
            'use_regime_short_buffers': model_config.pop('use_regime_short_buffers', False),
            'regime_short_mode': model_config.pop('regime_short_mode', 'hard'),
            'regime_short_tau': model_config.pop('regime_short_tau', 1.0),
            'use_regime_knn_penalty': model_config.pop('use_regime_knn_penalty', False),
            'regime_knn_lambda_long': model_config.pop('regime_knn_lambda_long', 0.0),
            'regime_knn_lambda_short': model_config.pop('regime_knn_lambda_short', 0.0),
            'regime_reweight_gamma_long': model_config.pop('regime_reweight_gamma_long', 0.0),
            'regime_reweight_gamma_short': model_config.pop('regime_reweight_gamma_short', 0.0),
            'use_regime_conf_temp': model_config.pop('use_regime_conf_temp', False),
            'regime_conf_temp_beta_long': model_config.pop('regime_conf_temp_beta_long', 0.0),
            'regime_conf_temp_beta_short': model_config.pop('regime_conf_temp_beta_short', 0.0),
            'use_regime_clip': model_config.pop('use_regime_clip', False),
            'regime_clip_min': model_config.pop('regime_clip_min', 0.8),
            'regime_clip_max': model_config.pop('regime_clip_max', 3.0),
            'use_drift_mixw': model_config.pop('use_drift_mixw', False),
            'use_drift_k': model_config.pop('use_drift_k', False),
            'drift_ema_alpha': model_config.pop('drift_ema_alpha', 0.01),
            'drift_threshold': model_config.pop('drift_threshold', 0.25),
            'drift_mixw_low': model_config.pop('drift_mixw_low', model_config.get('dual_buffer_mix_w', 0.2)),
            'drift_mixw_high': model_config.pop('drift_mixw_high', min(0.8, model_config.get('dual_buffer_mix_w', 0.2) + 0.1)),
            'drift_k_low': model_config.pop('drift_k_low', model_config.get('online_k', 20)),
            'drift_k_high': model_config.pop('drift_k_high', max(model_config.get('online_k', 20), 2 * model_config.get('online_k', 20))),
            'dual_buffer_short_softmax_temp': model_config.pop('dual_buffer_short_softmax_temp', model_config.get('knn_softmax_temp', 2.0)),
            'dual_buffer_long_softmax_temp': model_config.pop('dual_buffer_long_softmax_temp', model_config.get('knn_softmax_temp', 2.0)),
            'dual_buffer_short_use_raw': model_config.pop('dual_buffer_short_use_raw', False),
            # Surprise-driven short-buffer writes
            'short_write_strategy': model_config.pop('short_write_strategy', 'all'),
            'short_write_topm': model_config.pop('short_write_topm', 16),
            'short_write_quantile': model_config.pop('short_write_quantile', 0.9),
            'short_write_dedup_eps': model_config.pop('short_write_dedup_eps', 0.1),
            # Per-regime online_scale gating
            'use_regime_online_scale': model_config.pop('use_regime_online_scale', False),
            'fit_regime_online_scale': model_config.pop('fit_regime_online_scale', False),
            'regime_online_scale_ridge': model_config.pop('regime_online_scale_ridge', 1e-3),
            'regime_online_scale_clip': model_config.pop('regime_online_scale_clip', 2.0),
            # TabPFN auxiliary gating (does not replace slow prior)
            'use_tabpfn_aux': model_config.pop('use_tabpfn_aux', False),
            'tabpfn_aux_weight': model_config.pop('tabpfn_aux_weight', 0.0),
            'tabpfn_aux_clip': model_config.pop('tabpfn_aux_clip', 0.2),
            'tabpfn_aux_config': model_config.pop('tabpfn_aux_config', None),
            # Ensemble prior gating (Tier 2: sample-adaptive integration)
            'use_ensemble_gating': model_config.pop('use_ensemble_gating', False),
            'ensemble_variants': model_config.pop('ensemble_variants', None),
            'ensemble_mode': model_config.pop('ensemble_mode', 'fixed'),
            'ensemble_weights': model_config.pop('ensemble_weights', None),
            'ensemble_gating_hidden': model_config.pop('ensemble_gating_hidden', 128),
            'ensemble_gating_lr': model_config.pop('ensemble_gating_lr', 1e-3),
            'ensemble_gating_epochs': model_config.pop('ensemble_gating_epochs', 80),
            'ensemble_gating_wd': model_config.pop('ensemble_gating_wd', 2e-5),
            # Scheme C: Disagreement-driven correction scaling
            'use_disagree_scale': model_config.pop('use_disagree_scale', False),
            'disagree_strength': model_config.pop('disagree_strength', 1.0),
            'disagree_ema_alpha': model_config.pop('disagree_ema_alpha', 0.05),
            'disagree_scale_min': model_config.pop('disagree_scale_min', 0.5),
            'disagree_scale_max': model_config.pop('disagree_scale_max', 2.0),
            # Cascaded residual correction (Layer 2)
            'use_cascaded_residual': model_config.pop('use_cascaded_residual', False),
            'cascaded_scale': model_config.pop('cascaded_scale', 0.194),
            'cascaded_k': model_config.pop('cascaded_k', 47),
            'cascaded_temp': model_config.pop('cascaded_temp', 1.30),
            'cascaded_clip': model_config.pop('cascaded_clip', 2.67),
            'cascaded_buf_size': model_config.pop('cascaded_buf_size', 4096),
            'cascaded_warmup': model_config.pop('cascaded_warmup', 385),
            # Tree leaf embeddings for classification kNN
            'cls_embedding_type': model_config.pop('cls_embedding_type', 'limix'),
            'device': str(self.args.device),
        }

        # Pass remaining config as kwargs (will be consumed by **kwargs)
        config.update(model_config)

        self.model = ThreeTierPFC(**config).to(self.args.device)
        self.model.freeze_slow_prior()

        # Load tree leaf embeddings for classification kNN if configured
        dataset_name = getattr(self.args, 'dataset', None)
        if dataset_name:
            self.model.load_tree_leaf_embeddings(dataset_name)

    def _setup_slow_prior(self):
        """Setup slow prior context and pre-compute predictions/embeddings."""
        X_train = self.N['train']
        y_train = self.y['train']

        if isinstance(X_train, torch.Tensor):
            X_train = X_train.cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.cpu().numpy()

        # The pipeline standardises regression labels; tell the prior what statistics it
        # used, so a cache written in raw target units can be converted on load.
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            self.model.set_raw_target_stats(self.y_info['mean'], self.y_info['std'])

        self.model.set_slow_prior_context(X_train, y_train)

        cache_dir = getattr(self.args, 'save_path', None)
        dataset_name = getattr(self.args, 'dataset', None)

        train_indices = np.arange(len(X_train))
        self.model.precompute_slow_prior(X_train, train_indices, 'train',
                                          cache_dir=cache_dir, dataset_name=dataset_name)

        if 'val' in self.N:
            X_val = self.N['val']
            if isinstance(X_val, torch.Tensor):
                X_val = X_val.cpu().numpy()
            val_indices = np.arange(len(X_val))
            self.model.precompute_slow_prior(X_val, val_indices, 'val',
                                              cache_dir=cache_dir, dataset_name=dataset_name)

        # Load ensemble variant prior caches if enabled
        if self.model.use_ensemble_gating and dataset_name:
            ctx = self.model.slow_prior_config.get('context_size', 3000)
            self.model.load_ensemble_prior_caches(dataset_name, ctx)

    def fit(self, data, info, train=True, config=None, best_epoch=None):
        """
        Fit with bio-learning: single pass → fit regime detector.
        """
        from model.lib.data import Dataset_TS, TData_TS
        from torch.utils.data import DataLoader, Dataset

        N, C, M, y = data
        self.D = Dataset_TS(N=N, C=C, M=M, y=y, info=info)
        self.N, self.C, self.M, self.y = self.D.N, self.D.C, self.D.M, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = (
            self.D.is_binclass, self.D.is_multiclass, self.D.is_regression
        )
        self.n_num_features, self.n_cat_features = self.D.n_num_features, self.D.n_cat_features

        if config is not None:
            self.reset_stats_withconfig(config)

        self.data_format(is_train=True)
        self.args.t_mean = self.D.t_mean
        self.args.t_std = self.D.t_std

        # Dataset with indices for cache lookup
        class IndexedDataset(Dataset):
            def __init__(self, base_dataset):
                self.base_dataset = base_dataset

            def __len__(self):
                return len(self.base_dataset)

            def __getitem__(self, idx):
                X, M, y = self.base_dataset[idx]
                return X, M, y, idx

        trainset = TData_TS(self.is_regression, (self.N, self.C), self.M, self.y, self.y_info, 'train')
        indexed_trainset = IndexedDataset(trainset)

        self.train_loader = DataLoader(
            dataset=indexed_trainset,
            batch_size=min(self.args.batch_size * 4, 2048),
            shuffle=False,  # Preserve temporal order
            num_workers=0
        )

        self.construct_model()
        self._setup_slow_prior()

        if not train:
            return

        # =================================================================
        # Single pass: accumulate embeddings + errors, then fit regime detector
        # =================================================================
        print("\n" + "="*70)
        print("BIO-LEARNING: Single pass to fit regime detector")
        print("="*70)

        self.model.init_bio_learning()
        self.model.eval()  # No dropout/batchnorm effects needed

        tic = time.time()
        n_batches = len(self.train_loader)

        for i, (X, M, y, indices) in enumerate(self.train_loader, 1):
            X = X.to(self.args.device)
            M = M.to(self.args.device)
            y = y.to(self.args.device)
            indices = indices.to(self.args.device)

            with torch.no_grad():
                output = self.model(X, None, M, labels=y, indices=indices, split='train')

                slow_pred = self.model._teacher_pred
                self.model.accumulate_bio_statistics(
                    features=self.model._last_raw_embeddings,
                    labels=y,
                    predictions=output.unsqueeze(-1) if output.dim() == 1 else output,
                    slow_pred=slow_pred,
                )

            if i % 20 == 0 or i == n_batches:
                print(f"  Batch {i}/{n_batches}")

        bio_data = self.model.finalize_bio_learning()

        # Fit global online_scale multiplier on validation set (scalar)
        if bio_data is not None and self.model.fit_global_online_scale:
            all_embeddings_train, all_errors_train = bio_data
            self._fit_global_online_scale(all_embeddings_train, all_errors_train)

        # Fit per-regime online_scale on validation set
        if bio_data is not None and self.model.fit_regime_online_scale:
            all_embeddings_train, all_errors_train = bio_data
            self._fit_regime_online_scales(all_embeddings_train, all_errors_train)
        elif self.model.fit_regime_online_scale:
            print("Fit regime online scale requested, but bio_data is None (skipping)")

        if bio_data is not None:
            del bio_data

        elapsed = time.time() - tic
        print(f"\nBio-learning completed in {elapsed:.2f}s")

        # Validate
        self.validate(0)

        # Save model
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, 'epoch-last-{}.pth'.format(str(self.args.seed)))
        )

        print(f"Best validation result: {self.trlog['best_res']:.4f}")
        return elapsed

    def validate(self, epoch):
        """Validate the model."""
        self.model.eval()
        vl = Averager()
        all_preds = []
        all_targets = []

        for i, (X, M, y) in enumerate(self.val_loader, 1):
            X = X.to(self.args.device)
            M = M.to(self.args.device)
            y = y.to(self.args.device)

            batch_size = X.size(0)
            start_idx = (i - 1) * self.val_loader.batch_size
            indices = torch.arange(start_idx, start_idx + batch_size, device=self.args.device)

            with torch.no_grad():
                with cuda_autocast():
                    output = self.model(X, None, M, indices=indices, split='val')
                    loss = self.criterion(output, y)

            vl.add(loss.item())
            all_preds.append(output.cpu())
            all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        if np.isnan(all_preds).any():
            print(f"Warning: NaN in predictions, replacing with 0")
            all_preds = np.nan_to_num(all_preds, nan=0.0)

        if self.is_regression:
            from sklearn.metrics import mean_squared_error
            vres = mean_squared_error(all_targets, all_preds) ** 0.5
            measure = np.less_equal
        else:
            from sklearn.metrics import accuracy_score
            vres = 1 - accuracy_score(all_targets, all_preds.argmax(1))
            measure = np.greater_equal

        print(f'Epoch {epoch}, val, loss={vl.item():.4f} result={vres:.4f}')

        if measure(vres, self.trlog['best_res']) or epoch == 0:
            self.trlog['best_res'] = vres
            self.trlog['best_epoch'] = epoch
            torch.save(
                dict(params=self.model.state_dict()),
                osp.join(self.args.save_path, 'best-val-{}.pth'.format(str(self.args.seed)))
            )

    def _fit_global_online_scale(self, train_embeddings: np.ndarray, train_errors: np.ndarray):
        """Fit a global online_scale multiplier using validation set.

        We build a temporary kNN buffer from training embeddings/errors, compute val kNN
        estimates, then solve a closed-form ridge regression for a single scalar s:

            minimize ||(y - base) - s * (online_scale * knn_est)||^2 + ridge*s^2

        We then store it as self.model.online_scale_mult.
        """
        from model.models.three_tier_pfc_bio import OnlineKNNBuffer

        print("\n" + "="*70)
        print("FITTING GLOBAL ONLINE SCALE MULTIPLIER ON VALIDATION SET")
        print("="*70)

        tmp_buffer = OnlineKNNBuffer(
            max_size=self.model.online_buffer.max_size,
            device=str(self.args.device),
            temporal_decay=self.model.online_buffer.temporal_decay,
            knn_kernel=self.model.online_buffer.knn_kernel,
            knn_agg=self.model.online_buffer.knn_agg,
            knn_trim_q=self.model.online_buffer.knn_trim_q,
            knn_clip=self.model.online_buffer.knn_clip,
            knn_softmax_temp=self.model.online_buffer.knn_softmax_temp,
        )

        train_emb_t = torch.tensor(train_embeddings, dtype=torch.float32, device=self.args.device)
        train_err_t = torch.tensor(train_errors, dtype=torch.float32, device=self.args.device)
        knn_emb_train = self.model._whiten(train_emb_t)
        tmp_buffer.write(knn_emb_train, train_err_t)
        del train_emb_t, train_err_t, knn_emb_train

        val_slow_preds = []
        val_labels = []
        val_knn_estimates = []

        self.model.eval()
        for i, (X, M, y) in enumerate(self.val_loader, 1):
            X = X.to(self.args.device)
            M = M.to(self.args.device)
            y = y.to(self.args.device)

            batch_size = X.size(0)
            start_idx = (i - 1) * self.val_loader.batch_size
            indices = torch.arange(start_idx, start_idx + batch_size, device=self.args.device)

            with torch.no_grad():
                slow_result = self.model.slow_prior(X, indices=indices, split='val')
                slow_pred = slow_result.get('prediction', None)
                raw_emb = slow_result.get('raw_embeddings', None)
                if raw_emb is None:
                    raw_emb = slow_result['features']
                if slow_pred is not None and slow_pred.dim() > 1:
                    slow_pred = slow_pred.squeeze(-1)

                knn_emb = self.model._whiten(raw_emb.detach())
                knn_est = tmp_buffer.query(knn_emb, k=self.model.online_k)

            val_slow_preds.append(slow_pred.cpu())
            val_labels.append(y.cpu())
            val_knn_estimates.append(knn_est.cpu())

        slow = torch.cat(val_slow_preds, dim=0).numpy()
        y = torch.cat(val_labels, dim=0).numpy()
        est = torch.cat(val_knn_estimates, dim=0).numpy()

        # base prediction excludes any kNN correction; we also exclude regime bias on purpose
        base = slow
        residual = y - base

        x = (self.model.online_scale * est).astype('float64')
        r = residual.astype('float64')
        nonzero = np.abs(x) > 1e-12

        ridge = float(self.model.global_online_scale_ridge)
        clip = float(self.model.global_online_scale_clip)

        if nonzero.sum() < 2:
            s = 1.0
        else:
            xn = x[nonzero]
            rn = r[nonzero]
            s = float((xn @ rn) / (xn @ xn + ridge))
            s = max(0.0, min(clip, s))

        self.model.online_scale_mult = s
        print(f"Fitted global online_scale_mult={s:.4f} (ridge={ridge}, clip={clip})")
        print("="*70 + "\n")
        del tmp_buffer

    def _fit_regime_online_scales(self, train_embeddings: np.ndarray, train_errors: np.ndarray):
        """Fit per-regime online_scale using validation set.

        Populates a temporary kNN buffer from training data, queries val embeddings
        against it to get kNN estimates, then fits per-regime scales via closed-form
        ridge regression.
        """
        from model.models.three_tier_pfc_bio import OnlineKNNBuffer

        print("\n" + "="*70)
        print("FITTING PER-REGIME ONLINE SCALE ON VALIDATION SET")
        print("="*70)

        # Create temporary kNN buffer from training data
        tmp_buffer = OnlineKNNBuffer(
            max_size=self.model.online_buffer.max_size,
            device=str(self.args.device),
            temporal_decay=self.model.online_buffer.temporal_decay,
            knn_kernel=self.model.online_buffer.knn_kernel,
            knn_agg=self.model.online_buffer.knn_agg,
            knn_trim_q=self.model.online_buffer.knn_trim_q,
            knn_clip=self.model.online_buffer.knn_clip,
            knn_softmax_temp=self.model.online_buffer.knn_softmax_temp,
        )

        # Fill buffer with whitened train embeddings + errors
        train_emb_t = torch.tensor(train_embeddings, dtype=torch.float32, device=self.args.device)
        train_err_t = torch.tensor(train_errors, dtype=torch.float32, device=self.args.device)
        knn_emb_train = self.model._whiten(train_emb_t)
        tmp_buffer.write(knn_emb_train, train_err_t)
        del train_emb_t, train_err_t, knn_emb_train

        # Collect val predictions, embeddings, labels, and kNN estimates
        val_slow_preds = []
        val_embeddings = []
        val_labels = []
        val_knn_estimates = []

        self.model.eval()
        for i, (X, M, y) in enumerate(self.val_loader, 1):
            X = X.to(self.args.device)
            M = M.to(self.args.device)
            y = y.to(self.args.device)

            batch_size = X.size(0)
            start_idx = (i - 1) * self.val_loader.batch_size
            indices = torch.arange(start_idx, start_idx + batch_size, device=self.args.device)

            with torch.no_grad():
                slow_result = self.model.slow_prior(X, indices=indices, split='val')
                slow_pred = slow_result.get('prediction', None)
                raw_emb = slow_result.get('raw_embeddings', None)
                if raw_emb is None:
                    raw_emb = slow_result['features']
                if slow_pred is not None and slow_pred.dim() > 1:
                    slow_pred = slow_pred.squeeze(-1)

                # kNN estimate from training buffer
                knn_emb = self.model._whiten(raw_emb.detach())
                knn_est = tmp_buffer.query(knn_emb, k=self.model.online_k)

            val_slow_preds.append(slow_pred.cpu().numpy())
            val_embeddings.append(raw_emb.detach().cpu().numpy())
            val_labels.append(y.cpu().numpy())
            val_knn_estimates.append(knn_est.cpu().numpy())

        val_slow_preds = np.concatenate(val_slow_preds, axis=0)
        val_embeddings = np.concatenate(val_embeddings, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_knn_estimates = np.concatenate(val_knn_estimates, axis=0)

        # Fit per-regime scales
        self.model.regime_detector.fit_online_scales(
            embeddings=val_embeddings,
            slow_preds=val_slow_preds,
            true_labels=val_labels,
            knn_estimates=val_knn_estimates,
            global_online_scale=self.model.online_scale,
            use_regime_bias=self.model.use_regime_bias,
            ridge=self.model.regime_online_scale_ridge,
            clip=self.model.regime_online_scale_clip,
        )

        # Enable the feature now that scales are fitted
        self.model.use_regime_online_scale = True
        del tmp_buffer
        print("="*70 + "\n")

    def predict(self, data, info, model_name):
        """Predict on test data with online adaptation."""
        N, C, M, y = data
        self.data_format(is_train=False, N=N, C=C, M=M, y=y)

        cache_dir = getattr(self.args, 'save_path', None)
        dataset_name = getattr(self.args, 'dataset', None)

        if hasattr(self, 'N_test') and self.N_test is not None:
            X_test = self.N_test
            if isinstance(X_test, torch.Tensor):
                X_test = X_test.cpu().numpy()
            test_indices = np.arange(len(X_test))
            self.model.precompute_slow_prior(X_test, test_indices, 'test',
                                              cache_dir=cache_dir, dataset_name=dataset_name)

        self.model.load_state_dict(
            torch.load(osp.join(self.args.save_path, f'{model_name}-{self.args.seed}.pth'))['params']
        )
        self.model.eval()
        print(f'Best epoch {self.trlog["best_epoch"]}, best val res={self.trlog["best_res"]:.4f}')

        # Pass label std for denormalized gate metric
        if self.is_regression and self.y_info.get('policy') == 'mean_std':
            self.model.set_label_std(self.y_info['std'])

        # Enable online adaptation
        if self.use_online_adaptation:
            self.model.enable_online_adaptation(
                k=self.model.online_k,
                min_samples=self.model.online_min_samples,
                scale=self.model.online_scale,
            )

        # Optional: test-time self-calibration of kNN correction scale (chunk-level)
        calib_cfg = self.args.config.get('model', {})
        use_online_scale_calibration = calib_cfg.get('use_online_scale_calibration', False)
        calib_alpha = float(calib_cfg.get('calib_alpha', 0.05))
        calib_clip = float(calib_cfg.get('calib_clip', 2.0))
        calib_warmup_chunks = int(calib_cfg.get('calib_warmup_chunks', 4))
        online_scale_mult = 1.0
        calib_chunks_seen = 0

        test_logit = []
        test_label = []

        online_chunk_size = self.model.online_chunk_size if self.use_online_adaptation else None

        for i, (X, M, y) in enumerate(self.test_loader, 1):
            X = X.to(self.args.device)
            M = M.to(self.args.device)
            y = y.to(self.args.device)

            batch_size = X.size(0)
            start_idx = (i - 1) * self.test_loader.batch_size

            if online_chunk_size and self.use_online_adaptation:
                # Mini-batch processing for fine-grained online adaptation
                for chunk_start in range(0, batch_size, online_chunk_size):
                    chunk_end = min(chunk_start + online_chunk_size, batch_size)
                    X_chunk = X[chunk_start:chunk_end]
                    M_chunk = M[chunk_start:chunk_end]
                    y_chunk = y[chunk_start:chunk_end]
                    idx_chunk = torch.arange(
                        start_idx + chunk_start,
                        start_idx + chunk_end,
                        device=self.args.device
                    )

                    with torch.no_grad():
                        with cuda_autocast():
                            output_chunk = self.model(X_chunk, None, M_chunk,
                                                      indices=idx_chunk, split='test')

                    test_logit.append(output_chunk)
                    test_label.append(y_chunk)

                    # Optional: online scale calibration using chunk feedback
                    if use_online_scale_calibration:
                        with torch.no_grad():
                            raw_emb = self.model._last_raw_embeddings
                            slow_pred = self.model._teacher_pred
                            if raw_emb is not None and slow_pred is not None and self.model.online_buffer.current_size >= self.model.online_min_samples:
                                sp = slow_pred.detach()
                                # For classification, reduce to scalar for calibration
                                if not self.is_regression and sp.dim() > 1 and sp.shape[-1] > 1:
                                    sp = sp[:, 1] if sp.shape[-1] == 2 else sp.max(dim=-1).values
                                else:
                                    sp = sp.squeeze(-1) if sp.dim() > 1 else sp
                                # kNN estimate (residual) for this chunk
                                knn_emb = self.model._whiten(raw_emb.detach())
                                est = self.model.online_buffer.query(knn_emb, k=self.model.online_k)
                                e = (self.model.online_scale * est).detach()
                                r = (y_chunk.float() - sp).detach()
                                denom = (e * e).sum() + 1e-6
                                num = (e * r).sum()
                                s_hat = (num / denom).clamp(min=0.0, max=calib_clip).item()

                                if calib_chunks_seen >= calib_warmup_chunks:
                                    online_scale_mult = (1.0 - calib_alpha) * online_scale_mult + calib_alpha * float(s_hat)
                                calib_chunks_seen += 1
                                self.model.online_scale_mult = float(online_scale_mult)

                    # Update buffer with ground truth feedback
                    raw_emb = self.model._last_raw_embeddings
                    slow_pred = self.model._teacher_pred
                    if slow_pred is not None:
                        sp = slow_pred.detach()
                        with torch.no_grad():
                            self.model.update_online_buffer(raw_emb, sp, y_chunk)
                        # Online ensemble weight adaptation
                        self.model.update_ensemble_weights_online(y_chunk)
            else:
                indices = torch.arange(start_idx, start_idx + batch_size, device=self.args.device)

                with torch.no_grad():
                    with cuda_autocast():
                        output = self.model(X, None, M, indices=indices, split='test')

                test_logit.append(output)
                test_label.append(y)

        test_logit = torch.cat(test_logit, dim=0)
        test_label = torch.cat(test_label, dim=0)

        # Report stats
        if self.use_online_adaptation:
            stats = self.model.get_online_stats()
            print(f"Online adaptation stats: buffer_size={stats['online_buffer_size']}")
            if stats.get('use_regime_online_scale') and 'regime_online_scales' in stats:
                scales_str = [f"{s:.4f}" for s in stats['regime_online_scales']]
                print(f"  regime_online_scales: [{', '.join(scales_str)}]")
            self.model.disable_online_adaptation()

        vl = self.criterion(test_logit, test_label).item()
        vres, metric_name = self.metric(test_logit, test_label, self.y_info)

        print(f'Test: loss={vl:.4f}')
        for name, res in zip(metric_name, vres):
            print(f'[{name}]={res:.4f}')

        return vl, vres, metric_name, test_logit
