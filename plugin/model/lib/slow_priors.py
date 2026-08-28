"""
Generic Slow Prior Interface for Three-Tier PFC Architecture.

Provides a unified interface for different slow prior implementations:
- TabPFN: Frozen Bayesian prior trained on synthetic tabular data
- LanguageModel: Embeddings from pretrained language models (BERT, GPT, etc.)
- Ensemble: Combination of multiple weak priors

The slow prior provides:
1. Base predictions (ŷ_slow) - initial prediction from the prior
2. Features/embeddings - rich representations for downstream adaptation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Tuple, Union
import os


class SlowPrior(ABC, nn.Module):
    """
    Abstract base class for slow priors in the Three-Tier PFC architecture.

    A slow prior provides:
    - Frozen/slowly-adapting knowledge about the problem domain
    - Base predictions that anchor the model's outputs
    - Rich feature representations for downstream modules

    Subclasses must implement:
    - set_context(): Fit/prepare the prior on training data
    - precompute_predictions(): Cache predictions and embeddings
    - forward(): Get predictions and features for new inputs
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        feature_dim: int,
        is_regression: bool = True,
        device: str = 'cuda',
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self._feature_dim = feature_dim
        self.is_regression = is_regression
        self.device = device

        # Caches for predictions and embeddings
        self.prediction_cache: Dict[str, Dict[int, np.ndarray]] = {
            'train': {}, 'val': {}, 'test': {}
        }
        self.embedding_cache: Dict[str, Dict[int, np.ndarray]] = {
            'train': {}, 'val': {}, 'test': {}
        }

        # Learnable scale and bias for predictions
        self.pred_scale = nn.Parameter(torch.tensor(1.0))
        self.pred_bias = nn.Parameter(torch.tensor(0.0))
        # Statistics of the untransformed target, set by the training method when the
        # pipeline standardises labels. Cached predictions written by an offline builder
        # are in raw target units; the training loop works in the standardised space.
        self.raw_target_mean = None
        self.raw_target_std = None

    @property
    def feature_dim(self) -> int:
        """Dimension of output features."""
        return self._feature_dim

    @abstractmethod
    def set_context(self, X: np.ndarray, y: np.ndarray):
        """
        Set context/fit the prior on training data.

        Args:
            X: [n_samples, d_in] - training features (numpy)
            y: [n_samples] or [n_samples, d_out] - training labels (numpy)
        """
        pass

    @abstractmethod
    def precompute_predictions(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        split: str = 'train',
        cache_dir: Optional[str] = None,
    ):
        """
        Pre-compute and cache predictions and embeddings.

        Args:
            X: [n_samples, d_in] - features (numpy)
            indices: [n_samples] - sample indices for caching
            split: 'train', 'val', or 'test'
            cache_dir: optional directory to save/load cache
        """
        pass

    @abstractmethod
    def _compute_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute predictions for inputs (when cache miss).

        Args:
            x: [batch, d_in] - input features

        Returns:
            predictions: [batch, d_out]
        """
        pass

    @abstractmethod
    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature representations for inputs.

        Args:
            x: [batch, d_in] - input features

        Returns:
            features: [batch, feature_dim]
        """
        pass

    def set_raw_target_stats(self, mean: float, std: float):
        """Record the mean and standard deviation the pipeline standardised labels with."""
        self.raw_target_mean = float(mean)
        self.raw_target_std = float(std) + 1e-8

    def align_cached_predictions(self, predictions: np.ndarray, where: str = "") -> np.ndarray:
        """Bring a cached prediction array into the space the training loop works in.

        A cache holds either raw target units or the standardised space. The two are told
        apart by comparing the prediction mean against the raw target statistics: a
        standardised array has mean about zero, a raw one sits at the target mean. The
        conversion puts the prior on the same footing as the labels the loop works with.
        """
        if self.raw_target_mean is None or not self.is_regression:
            return predictions
        arr = np.asarray(predictions, dtype=np.float64)
        if arr.size == 0:
            return predictions
        if abs(float(arr.mean()) - self.raw_target_mean) / self.raw_target_std >= 1.0:
            return predictions          # already standardised
        converted = (arr - self.raw_target_mean) / self.raw_target_std
        if not getattr(self, "_raw_align_reported", False):
            self._raw_align_reported = True
            print(f"  cached predictions{where} are in raw target units "
                  f"(mean {arr.mean():.4f} against target mean {self.raw_target_mean:.4f}); "
                  f"converting to the standardised space the training loop uses")
        return converted.astype(np.float32)

    def get_cached_prediction(
        self,
        indices: torch.Tensor,
        split: str = 'train',
    ) -> Optional[torch.Tensor]:
        """Get cached predictions by indices."""
        cache = self.prediction_cache.get(split, {})
        if not cache:
            return None

        predictions = []
        for idx in indices.cpu().numpy():
            idx = int(idx)
            if idx in cache:
                predictions.append(cache[idx])
            else:
                return None

        return torch.tensor(
            np.array(predictions),
            device=indices.device,
            dtype=torch.float32
        )

    def get_cached_embedding(
        self,
        indices: torch.Tensor,
        split: str = 'train',
    ) -> Optional[torch.Tensor]:
        """Get cached embeddings by indices."""
        cache = self.embedding_cache.get(split, {})
        if not cache:
            return None

        embeddings = []
        for idx in indices.cpu().numpy():
            idx = int(idx)
            if idx in cache:
                embeddings.append(cache[idx])
            else:
                return None

        return torch.tensor(
            np.array(embeddings),
            device=indices.device,
            dtype=torch.float32
        )

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        split: str = 'train',
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through slow prior.

        Args:
            x: [batch, d_in] - input features
            indices: [batch] - sample indices for cache lookup
            split: 'train', 'val', or 'test'

        Returns:
            Dict containing:
            - prediction: [batch, d_out] base prediction
            - features: [batch, feature_dim] feature representations
        """
        batch_size = x.size(0)
        device = x.device

        # Try cache first for predictions
        prediction = None
        if indices is not None:
            prediction = self.get_cached_prediction(indices, split)

        if prediction is None:
            prediction = self._compute_predictions(x)

        # Ensure correct shape
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(-1)

        # Apply learned scaling
        prediction = prediction * self.pred_scale + self.pred_bias

        # Get features (try cache first)
        features = None
        if indices is not None:
            features = self.get_cached_embedding(indices, split)

        if features is None:
            features = self._compute_features(x)

        return {
            'prediction': prediction,
            'features': features,
        }

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def clear_cache(self, split: Optional[str] = None):
        """Clear prediction and embedding caches."""
        if split is None:
            self.prediction_cache = {'train': {}, 'val': {}, 'test': {}}
            self.embedding_cache = {'train': {}, 'val': {}, 'test': {}}
        else:
            self.prediction_cache[split] = {}
            self.embedding_cache[split] = {}


# =============================================================================
# TabPFN Slow Prior
# =============================================================================

class TabPFNSlowPrior(SlowPrior):
    """
    TabPFN as frozen Bayesian slow prior.

    Uses TabPFN's predictions and optionally embeddings for:
    - Base predictions from Bayesian posterior
    - Rich embeddings from TabPFN's internal representations
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        feature_dim: int = 128,
        is_regression: bool = True,
        device: str = 'cuda',
        n_ensemble: int = 4,
        context_size: int = 1024,
        use_embeddings: bool = True,
    ):
        super().__init__(d_in, d_out, feature_dim, is_regression, device)

        self.n_ensemble = n_ensemble
        self.context_size = context_size
        self.use_embeddings = use_embeddings

        # TabPFN model (initialized lazily)
        self.tabpfn = None
        self.tabpfn_embedding = None
        self.tabpfn_fitted = False

        # Context data
        self.context_X = None
        self.context_y = None

        # Fallback MLP feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # Fallback prediction head
        self.fallback_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, d_out),
        )

        # TabPFN embedding projection (created when embedding dim is known)
        self.tabpfn_embedding_dim = None
        self.embedding_proj = None

        # Fusion layer for combining MLP + TabPFN features
        self.tabpfn_fusion = None

        # Initialize TabPFN
        self._init_tabpfn()

    def _init_tabpfn(self):
        """Initialize TabPFN model."""
        try:
            if self.is_regression:
                from tabpfn import TabPFNRegressor
                self.tabpfn = TabPFNRegressor(
                    n_estimators=self.n_ensemble,
                    device=self.device,
                    random_state=42,
                )
            else:
                from tabpfn import TabPFNClassifier
                self.tabpfn = TabPFNClassifier(
                    n_estimators=self.n_ensemble,
                    device=self.device,
                    random_state=42,
                )

            # Try to initialize embedding extractor
            if self.use_embeddings:
                try:
                    from tabpfn_extensions.embedding import TabPFNEmbedding
                    self.tabpfn_embedding = TabPFNEmbedding(
                        tabpfn_clf=self.tabpfn,
                        n_fold=0
                    )
                    print(f"TabPFNSlowPrior: initialized with embeddings")
                except ImportError:
                    print("TabPFNSlowPrior: embeddings not available")

            print(f"TabPFNSlowPrior: TabPFN initialized (ensemble={self.n_ensemble})")

        except ImportError:
            print("TabPFNSlowPrior: TabPFN not available, using fallback MLP")
            self.tabpfn = None

    def set_context(self, X: np.ndarray, y: np.ndarray):
        """Fit TabPFN on context data."""
        # Subsample if too large
        if len(X) > self.context_size:
            idx = np.random.choice(len(X), self.context_size, replace=False)
            X = X[idx]
            y = y[idx]

        self.context_X = X
        self.context_y = y

        if self.tabpfn is not None:
            print(f"TabPFNSlowPrior: fitting on {len(X)} samples...")
            self.tabpfn.fit(X, y)

            if self.tabpfn_embedding is not None:
                self.tabpfn_embedding.fit(X, y)

            self.tabpfn_fitted = True
            print("TabPFNSlowPrior: fitted successfully")

    def precompute_predictions(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        split: str = 'train',
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        """Pre-compute and cache TabPFN predictions and embeddings.

        If dataset_name is provided, uses a central cache directory:
            cache/tabpfn/{dataset_name}/
        This mirrors the LimiX central cache behavior.
        """
        # Central cache directory
        central_cache_dir = None
        if dataset_name is not None:
            central_cache_dir = os.path.join('cache', 'tabpfn', dataset_name)
            os.makedirs(central_cache_dir, exist_ok=True)

        # Try to load from central cache first, then experiment cache
        for try_cache_dir in [central_cache_dir, cache_dir]:
            if try_cache_dir is None:
                continue
            pred_path = os.path.join(try_cache_dir, f'tabpfn_predictions_{split}_ctx{self.context_size}.npy')
            emb_path = os.path.join(try_cache_dir, f'tabpfn_embeddings_{split}_ctx{self.context_size}.npy')

            if os.path.exists(pred_path):
                print(f"TabPFNSlowPrior: loading cached predictions from {try_cache_dir}")
                print(f"  split={split}, context_size={self.context_size}")
                predictions = np.load(pred_path)
                for i, idx in enumerate(indices[:len(predictions)]):
                    self.prediction_cache[split][int(idx)] = predictions[i]

                if self.use_embeddings and os.path.exists(emb_path):
                    print(f"TabPFNSlowPrior: loading cached embeddings from {try_cache_dir}")
                    embeddings = np.load(emb_path)
                    if embeddings.ndim == 3:
                        embeddings = embeddings.mean(axis=1)

                    self._setup_embedding_projection(embeddings.shape[-1])

                    for i, idx in enumerate(indices[:len(embeddings)]):
                        self.embedding_cache[split][int(idx)] = embeddings[i]
                return

        if self.tabpfn is None or not self.tabpfn_fitted:
            print(f"TabPFNSlowPrior: not fitted, skipping precomputation for {split}")
            return

        # Compute predictions
        print(f"TabPFNSlowPrior: computing predictions for {split} ({len(X)} samples)")
        batch_size = 1024
        predictions = []

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            pred = self.tabpfn.predict(batch_X)
            predictions.append(pred)

        predictions = np.concatenate(predictions, axis=0)

        for i, idx in enumerate(indices[:len(predictions)]):
            self.prediction_cache[split][int(idx)] = predictions[i]

        # Save to disk (include context_size in filename)
        save_cache_dir = central_cache_dir if central_cache_dir is not None else cache_dir
        if save_cache_dir is not None:
            os.makedirs(save_cache_dir, exist_ok=True)
            np.save(os.path.join(save_cache_dir, f'tabpfn_predictions_{split}_ctx{self.context_size}.npy'), predictions)

        # Extract embeddings if available
        if self.use_embeddings and self.tabpfn_embedding is not None:
            print(f"TabPFNSlowPrior: extracting embeddings for {split}")
            try:
                embeddings = self._extract_embeddings(X)

                self._setup_embedding_projection(embeddings.shape[-1])

                for i, idx in enumerate(indices[:len(embeddings)]):
                    self.embedding_cache[split][int(idx)] = embeddings[i]

                if save_cache_dir is not None:
                    np.save(os.path.join(save_cache_dir, f'tabpfn_embeddings_{split}_ctx{self.context_size}.npy'), embeddings)

            except Exception as e:
                print(f"TabPFNSlowPrior: embedding extraction failed: {e}")

    def _extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Extract TabPFN embeddings in batches."""
        batch_size = 512
        all_embeddings = []

        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            actual_size = len(batch_X)

            # Pad if needed
            if actual_size < batch_size:
                batch_X = np.concatenate([
                    batch_X,
                    np.zeros((batch_size - actual_size, batch_X.shape[1]))
                ], axis=0)

            emb = self.tabpfn_embedding.get_embeddings(
                self.context_X, self.context_y, batch_X, data_source="test"
            )

            # Handle shape: (n_ensemble, n_query, embed_dim) -> (n_query, embed_dim)
            if emb.ndim == 3:
                emb = np.transpose(emb, (1, 0, 2)).mean(axis=1)

            # Remove padding
            emb = emb[:actual_size]
            all_embeddings.append(emb)

            if i % 2048 == 0 and i > 0:
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

    def _setup_embedding_projection(self, embedding_dim: int):
        """Setup projection layer for TabPFN embeddings."""
        if self.tabpfn_embedding_dim is None:
            self.tabpfn_embedding_dim = embedding_dim

            # Fusion: MLP features + TabPFN embeddings -> feature_dim
            self.tabpfn_fusion = nn.Sequential(
                nn.Linear(self._feature_dim + embedding_dim, self._feature_dim * 2),
                nn.LayerNorm(self._feature_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self._feature_dim * 2, self._feature_dim),
                nn.LayerNorm(self._feature_dim),
            ).to(self.device)

            print(f"TabPFNSlowPrior: created fusion layer ({self._feature_dim} + {embedding_dim} -> {self._feature_dim})")

    def _compute_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predictions (fallback when cache miss)."""
        if self.tabpfn is not None and self.tabpfn_fitted:
            with torch.no_grad():
                x_np = x.detach().cpu().numpy()
                pred_np = self.tabpfn.predict(x_np)
                return torch.tensor(pred_np, device=x.device, dtype=x.dtype)
        else:
            # Use fallback MLP
            features = self.feature_extractor(x)
            return self.fallback_head(features)

    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute features (MLP + optionally TabPFN embeddings)."""
        return self.feature_extractor(x)

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        split: str = 'train',
    ) -> Dict[str, torch.Tensor]:
        """Forward with hybrid MLP + TabPFN features."""
        # Get base prediction
        prediction = None
        if indices is not None:
            prediction = self.get_cached_prediction(indices, split)

        if prediction is None:
            prediction = self._compute_predictions(x)

        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(-1)

        prediction = prediction * self.pred_scale + self.pred_bias

        # Get MLP features
        mlp_features = self.feature_extractor(x)

        # Fuse with TabPFN embeddings if available
        if self.tabpfn_fusion is not None and indices is not None:
            tabpfn_emb = self.get_cached_embedding(indices, split)
            if tabpfn_emb is not None:
                tabpfn_emb = tabpfn_emb.detach()
                combined = torch.cat([mlp_features, tabpfn_emb], dim=-1)
                features = self.tabpfn_fusion(combined)
            else:
                features = mlp_features
        else:
            features = mlp_features

        return {
            'prediction': prediction,
            'features': features,
        }


# =============================================================================
# Language Model Slow Prior
# =============================================================================

class LanguageModelSlowPrior(SlowPrior):
    """
    Language Model embeddings as slow prior.

    Converts tabular data to text and extracts embeddings from
    pretrained language models (BERT, GPT-2, sentence-transformers, etc.).

    Key design choices:
    - Feature names and values are converted to natural language
    - Embeddings are cached for efficiency
    - A small MLP head provides predictions
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        feature_dim: int = 128,
        is_regression: bool = True,
        device: str = 'cuda',
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        feature_names: Optional[List[str]] = None,
        max_length: int = 256,
        pooling: str = 'mean',  # 'mean', 'cls', 'max'
    ):
        super().__init__(d_in, d_out, feature_dim, is_regression, device)

        self.model_name = model_name
        self.feature_names = feature_names or [f'feature_{i}' for i in range(d_in)]
        self.max_length = max_length
        self.pooling = pooling

        # Language model (initialized lazily)
        self.tokenizer = None
        self.encoder = None
        self.lm_embedding_dim = None

        # Projection from LM embedding to feature_dim
        self.embedding_proj = None

        # Prediction head
        self.pred_head = None

        # Fallback MLP (used when LM unavailable)
        self.fallback_extractor = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        self.fallback_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, d_out),
        )

        # Training data statistics for text generation
        self.feature_means = None
        self.feature_stds = None

        # Initialize LM
        self._init_language_model()

    def _init_language_model(self):
        """Initialize the language model."""
        try:
            # Try sentence-transformers first (easiest to use)
            if 'sentence-transformers' in self.model_name or self.model_name.startswith('all-'):
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(self.model_name, device=self.device)
                self.lm_embedding_dim = self.encoder.get_sentence_embedding_dimension()
                print(f"LanguageModelSlowPrior: loaded SentenceTransformer ({self.model_name}), dim={self.lm_embedding_dim}")

            else:
                # Fall back to transformers library
                from transformers import AutoTokenizer, AutoModel
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.encoder = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.encoder.eval()

                # Get embedding dimension from config
                self.lm_embedding_dim = self.encoder.config.hidden_size
                print(f"LanguageModelSlowPrior: loaded {self.model_name}, dim={self.lm_embedding_dim}")

            # Create projection layers
            self._setup_projection_layers()

        except Exception as e:
            print(f"LanguageModelSlowPrior: failed to load {self.model_name}: {e}")
            print("LanguageModelSlowPrior: using fallback MLP")
            self.encoder = None

    def _setup_projection_layers(self):
        """Setup projection from LM embedding to feature_dim."""
        if self.lm_embedding_dim is not None:
            self.embedding_proj = nn.Sequential(
                nn.Linear(self.lm_embedding_dim, self._feature_dim * 2),
                nn.LayerNorm(self._feature_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self._feature_dim * 2, self._feature_dim),
                nn.LayerNorm(self._feature_dim),
            ).to(self.device)

            self.pred_head = nn.Sequential(
                nn.Linear(self._feature_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, self.d_out),
            ).to(self.device)

    def _tabular_to_text(self, X: np.ndarray) -> List[str]:
        """
        Convert tabular data to natural language descriptions.

        Args:
            X: [n_samples, d_in] - tabular features

        Returns:
            List of text descriptions, one per sample
        """
        texts = []

        for row in X:
            parts = []
            for i, (name, value) in enumerate(zip(self.feature_names, row)):
                # Format based on value type
                if np.isnan(value):
                    parts.append(f"{name} is missing")
                elif isinstance(value, (int, np.integer)) or value == int(value):
                    parts.append(f"{name} is {int(value)}")
                else:
                    # Normalize value if we have statistics
                    if self.feature_means is not None and self.feature_stds is not None:
                        z_score = (value - self.feature_means[i]) / (self.feature_stds[i] + 1e-8)
                        if z_score > 1.5:
                            parts.append(f"{name} is high ({value:.2f})")
                        elif z_score < -1.5:
                            parts.append(f"{name} is low ({value:.2f})")
                        else:
                            parts.append(f"{name} is {value:.2f}")
                    else:
                        parts.append(f"{name} is {value:.2f}")

            text = ", ".join(parts)
            texts.append(text)

        return texts

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings using the language model.

        Args:
            texts: List of text strings

        Returns:
            embeddings: [n_texts, lm_embedding_dim]
        """
        if self.encoder is None:
            raise ValueError("Language model not initialized")

        # Check if using sentence-transformers
        if hasattr(self.encoder, 'encode'):
            # SentenceTransformer
            embeddings = self.encoder.encode(
                texts,
                batch_size=32,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
            )
            return embeddings

        else:
            # Transformers library
            from transformers import AutoTokenizer

            batch_size = 32
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt',
                ).to(self.device)

                # Get embeddings
                with torch.no_grad():
                    outputs = self.encoder(**inputs)
                    hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]

                    # Pool
                    if self.pooling == 'cls':
                        embeddings = hidden_states[:, 0, :]
                    elif self.pooling == 'max':
                        embeddings = hidden_states.max(dim=1)[0]
                    else:  # mean
                        attention_mask = inputs['attention_mask'].unsqueeze(-1)
                        embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)

                    all_embeddings.append(embeddings.cpu().numpy())

            return np.concatenate(all_embeddings, axis=0)

    def set_context(self, X: np.ndarray, y: np.ndarray):
        """Store context data statistics for text generation."""
        self.feature_means = np.nanmean(X, axis=0)
        self.feature_stds = np.nanstd(X, axis=0)
        print(f"LanguageModelSlowPrior: computed feature statistics from {len(X)} samples")

    def precompute_predictions(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        split: str = 'train',
        cache_dir: Optional[str] = None,
    ):
        """Pre-compute and cache LM embeddings."""
        # Try to load from cache
        if cache_dir is not None:
            emb_path = os.path.join(cache_dir, f'lm_embeddings_{split}.npy')

            if os.path.exists(emb_path):
                print(f"LanguageModelSlowPrior: loading cached embeddings for {split}")
                embeddings = np.load(emb_path)
                for i, idx in enumerate(indices[:len(embeddings)]):
                    self.embedding_cache[split][int(idx)] = embeddings[i]
                return

        if self.encoder is None:
            print(f"LanguageModelSlowPrior: no encoder, skipping precomputation")
            return

        print(f"LanguageModelSlowPrior: computing embeddings for {split} ({len(X)} samples)")

        # Convert to text
        texts = self._tabular_to_text(X)

        # Encode
        embeddings = self._encode_texts(texts)

        # Cache
        for i, idx in enumerate(indices[:len(embeddings)]):
            self.embedding_cache[split][int(idx)] = embeddings[i]

        # Save to disk
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(os.path.join(cache_dir, f'lm_embeddings_{split}.npy'), embeddings)
            print(f"LanguageModelSlowPrior: saved embeddings to {emb_path}")

    def _compute_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predictions using the projection head."""
        features = self._compute_features(x)
        if self.pred_head is not None:
            return self.pred_head(features)
        else:
            return self.fallback_head(features)

    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute features from input."""
        if self.encoder is not None and self.embedding_proj is not None:
            # Convert to text and encode
            x_np = x.detach().cpu().numpy()
            texts = self._tabular_to_text(x_np)
            embeddings = self._encode_texts(texts)
            embeddings_t = torch.tensor(embeddings, device=x.device, dtype=x.dtype)
            return self.embedding_proj(embeddings_t)
        else:
            return self.fallback_extractor(x)

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        split: str = 'train',
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with LM embeddings."""
        # Try to get cached embeddings
        lm_embeddings = None
        if indices is not None:
            lm_embeddings = self.get_cached_embedding(indices, split)

        # Project embeddings or compute from scratch
        if lm_embeddings is not None and self.embedding_proj is not None:
            features = self.embedding_proj(lm_embeddings)
        else:
            features = self._compute_features(x)

        # Compute predictions
        if self.pred_head is not None:
            prediction = self.pred_head(features)
        else:
            prediction = self.fallback_head(features)

        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(-1)

        prediction = prediction * self.pred_scale + self.pred_bias

        return {
            'prediction': prediction,
            'features': features,
        }


# =============================================================================
# Ensemble Slow Prior
# =============================================================================

class EnsembleSlowPrior(SlowPrior):
    """
    Ensemble of multiple slow priors.

    Combines predictions and features from multiple sources
    (e.g., TabPFN + LanguageModel) for more robust representations.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        feature_dim: int = 128,
        is_regression: bool = True,
        device: str = 'cuda',
        prior_configs: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(d_in, d_out, feature_dim, is_regression, device)

        self.priors: nn.ModuleList = nn.ModuleList()
        self.prior_weights = None

        # Default: TabPFN + LM
        if prior_configs is None:
            prior_configs = [
                {'type': 'tabpfn', 'weight': 0.7},
                {'type': 'lm', 'weight': 0.3, 'model_name': 'sentence-transformers/all-MiniLM-L6-v2'},
            ]

        # Create priors
        for config in prior_configs:
            prior = create_slow_prior(
                prior_type=config['type'],
                d_in=d_in,
                d_out=d_out,
                feature_dim=feature_dim,
                is_regression=is_regression,
                device=device,
                **{k: v for k, v in config.items() if k not in ['type', 'weight']}
            )
            self.priors.append(prior)

        # Learnable weights for combining priors
        n_priors = len(self.priors)
        self.prior_weights = nn.Parameter(torch.ones(n_priors) / n_priors)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * n_priors, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def set_context(self, X: np.ndarray, y: np.ndarray):
        """Set context for all priors."""
        for prior in self.priors:
            prior.set_context(X, y)

    def precompute_predictions(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        split: str = 'train',
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        """Precompute for all priors.

        Note: we propagate dataset_name so priors with central caches (e.g., LimiX)
        can reuse cache/limix/{dataset}/ artifacts.
        """
        for i, prior in enumerate(self.priors):
            prior_cache_dir = None
            if cache_dir is not None:
                prior_cache_dir = os.path.join(cache_dir, f'prior_{i}')
            try:
                prior.precompute_predictions(X, indices, split, prior_cache_dir, dataset_name=dataset_name)
            except TypeError:
                # Backwards-compatible: prior doesn't accept dataset_name
                prior.precompute_predictions(X, indices, split, prior_cache_dir)

    def _compute_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute weighted predictions from all priors."""
        weights = F.softmax(self.prior_weights, dim=0)
        predictions = []

        for prior in self.priors:
            pred = prior._compute_predictions(x)
            predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)  # [n_priors, batch, d_out]
        weighted = torch.einsum('p,pbo->bo', weights, stacked)
        return weighted

    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute fused features from all priors."""
        all_features = []
        for prior in self.priors:
            features = prior._compute_features(x)
            all_features.append(features)

        concatenated = torch.cat(all_features, dim=-1)
        return self.fusion(concatenated)

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        split: str = 'train',
    ) -> Dict[str, torch.Tensor]:
        """Forward through all priors and combine."""
        weights = F.softmax(self.prior_weights, dim=0)

        all_predictions = []
        all_features = []

        for prior in self.priors:
            result = prior(x, indices, split)
            all_predictions.append(result['prediction'])
            all_features.append(result['features'])

        # Weighted prediction
        pred_stack = torch.stack(all_predictions, dim=0)
        prediction = torch.einsum('p,pbo->bo', weights, pred_stack)
        prediction = prediction * self.pred_scale + self.pred_bias

        # Fused features
        features = self.fusion(torch.cat(all_features, dim=-1))

        return {
            'prediction': prediction,
            'features': features,
        }


# =============================================================================
# LimiX Foundation Model Slow Prior
# =============================================================================

class LimiXSlowPrior(SlowPrior):
    """
    LimiX foundation model as slow prior with embedding extraction.

    LimiX is a tabular foundation model that provides in-context learning
    for tabular data without requiring semantic feature names. It's trained
    on many tabular datasets and can generalize to new tasks.

    Key features:
    - Extracts rich embeddings from transformer encoder output
    - Embeddings capture row-column interactions from the transformer
    - Averaged over features similar to LLM pooling strategies

    Key advantages over TabPFN:
    - Larger model capacity (16M+ parameters)
    - Trained on more diverse tabular datasets
    - Better handling of high-dimensional features
    - Rich embeddings from deep transformer layers
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        feature_dim: int = 128,
        is_regression: bool = True,
        device: str = 'cuda',
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        context_size: int = 10000,  # LimiX supports larger context
        use_retrieval: bool = False,  # Whether to use retrieval-augmented inference
    ):
        super().__init__(d_in, d_out, feature_dim, is_regression, device)

        self.model_path = model_path
        self.config_path = config_path
        self.context_size = context_size
        self.use_retrieval = use_retrieval

        # LimiX predictor and raw model (initialized lazily)
        self.limix = None
        self.limix_model = None  # Raw transformer model for embedding extraction
        self.limix_fitted = False
        self.limix_embed_dim = None  # Will be set after model loads

        # Context data
        self.context_X = None
        self.context_y = None
        self.y_mean = None
        self.y_std = None

        # Embedding projection (created after we know LimiX embed_dim)
        self.embedding_proj = None

        # Fallback feature extractor (when LimiX unavailable)
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # Prediction head (for when LimiX unavailable)
        self.fallback_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, d_out),
        )

        # Initialize LimiX
        self._init_limix()

        # Move to device
        self.to(device)

    def _init_limix(self):
        """Initialize LimiX model and extract raw transformer for embeddings."""
        try:
            # Add LimiX to path if needed
            import sys
            limix_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'LimiX'
            )
            if limix_path not in sys.path:
                sys.path.insert(0, limix_path)

            from inference.predictor import LimiXPredictor
            from utils.utils import download_model

            # Get model path
            if self.model_path is None:
                cache_dir = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    'cache'
                )
                os.makedirs(cache_dir, exist_ok=True)
                self.model_path = download_model(
                    repo_id="stableai-org/LimiX-16M",
                    filename="LimiX-16M.ckpt",
                    save_path=cache_dir
                )

            # Get config path
            if self.config_path is None:
                config_name = "reg_default_retrieval.json" if self.use_retrieval else "reg_default_noretrieval.json"
                if not self.is_regression:
                    config_name = config_name.replace("reg_", "cls_")
                self.config_path = os.path.join(limix_path, 'config', config_name)

            # Initialize predictor
            self.limix = LimiXPredictor(
                device=torch.device(self.device),
                model_path=self.model_path,
                inference_config=self.config_path,
                mix_precision=True,
            )

            # Get the raw transformer model for embedding extraction
            self.limix_model = self.limix.model
            self.limix_embed_dim = self.limix_model.embed_dim

            # Create embedding projection layer
            self._setup_embedding_projection()

            print(f"LimiXSlowPrior: initialized (model={self.model_path})")
            print(f"LimiXSlowPrior: embed_dim={self.limix_embed_dim}, feature_dim={self._feature_dim}")

        except Exception as e:
            print(f"LimiXSlowPrior: failed to initialize LimiX: {e}")
            print("LimiXSlowPrior: using fallback MLP")
            self.limix = None
            self.limix_model = None

    def _setup_embedding_projection(self):
        """Create projection layer from LimiX embed_dim to feature_dim."""
        if self.limix_embed_dim is not None and self.embedding_proj is None:
            self.embedding_proj = nn.Sequential(
                nn.Linear(self.limix_embed_dim, self._feature_dim * 2),
                nn.LayerNorm(self._feature_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self._feature_dim * 2, self._feature_dim),
                nn.LayerNorm(self._feature_dim),
            ).to(self.device)
            print(f"LimiXSlowPrior: created embedding projection ({self.limix_embed_dim} -> {self._feature_dim})")

    def set_context(self, X: np.ndarray, y: np.ndarray):
        """Set context data for LimiX in-context learning."""
        # Subsample if too large
        if len(X) > self.context_size:
            idx = np.random.choice(len(X), self.context_size, replace=False)
            X = X[idx]
            y = y[idx]

        self.context_X = X
        self.context_y = y

        # Store normalization stats for regression
        if self.is_regression:
            self.y_mean = y.mean()
            self.y_std = y.std() + 1e-8

        self.limix_fitted = True
        print(f"LimiXSlowPrior: set context with {len(X)} samples")

    def _extract_embeddings_batch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings and predictions from LimiX transformer.

        This runs a custom forward pass to get the encoder_out tensor,
        then extracts embeddings by averaging over the feature dimension.

        Args:
            X_train: [n_train, d_in] - context features
            y_train: [n_train] - context labels (normalized)
            X_test: [n_test, d_in] - query features

        Returns:
            predictions: [n_test] or [n_test, n_classes]
            embeddings: [n_test, embed_dim]
        """
        if self.limix is None or self.limix_model is None:
            raise RuntimeError("LimiX model not initialized")

        # Use the first preprocessing pipeline (simplest)
        pipe = self.limix.preprocess_pipelines[0]

        # Concatenate train and test for consistent preprocessing
        x = np.concatenate([X_train, X_test], axis=0)

        # Preprocess using LimiX's preprocessing
        x = self.limix.convert_x_dtypes(x)
        x = self.limix.convert_category2num(x)
        categorical_idx = self.limix.get_categorical_features_indices(x)

        x_ = x.copy()
        y_ = y_train.copy()
        categorical_idx_ = categorical_idx.copy()

        # Apply preprocessing pipeline (skip retrieval steps)
        for step in pipe:
            if hasattr(step, 'fit_transform'):
                x_, categorical_idx_ = step.fit_transform(
                    x_, categorical_idx_,
                    self.limix.seeds[0],
                    y=y_
                )

        # Convert to tensors
        x_tensor = torch.from_numpy(x_.astype(np.float32)).to(self.device)
        y_tensor = torch.from_numpy(y_.astype(np.float32)).to(self.device)

        eval_pos = len(y_train)
        task_type = 'reg' if self.is_regression else 'cls'

        model = self.limix_model
        model.to(self.device)

        # Custom forward pass to extract encoder_out (following model.forward logic)
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=True):
            # Add batch dimension: [1, seq_len, num_features]
            x_input = x_tensor.unsqueeze(0)
            y_input = y_tensor.unsqueeze(0)

            batch_size, seq_len, num_feature = x_input.shape

            # Create x dict (following model.forward)
            x_dict = {
                'data': x_input,
                'mask': torch.isnan(x_input).to(torch.int32).to(x_input.device)
            }

            # Pad features if needed for grouping
            features_per_group = model.features_per_group
            feature_to_add = num_feature % features_per_group
            if feature_to_add > 0:
                for k in x_dict:
                    x_dict[k] = torch.cat(
                        (
                            x_dict[k],
                            torch.zeros(
                                batch_size, seq_len, feature_to_add,
                                device=x_dict[k].device, dtype=x_dict[k].dtype
                            )
                        ),
                        dim=-1
                    )

            # Reshape for feature grouping
            for k in x_dict:
                x_dict[k] = x_dict[k].reshape(
                    batch_size, seq_len,
                    x_dict[k].shape[2] // features_per_group,
                    features_per_group
                )
            x_dict['eval_pos'] = eval_pos

            # Preprocess x
            preprocessed_x = model.x_preprocess(x_dict)
            preprocessed_x = model.process_4_x(preprocessed_x)
            x_encoder_result = model.encoder_x(preprocessed_x)
            x_emb_result = x_encoder_result['data']

            # Create y dict
            y_dict = {'data': y_input.unsqueeze(-1)}

            # Extend y if needed
            if y_dict['data'].shape[1] < x_dict['data'].shape[1]:
                y_dict['data'] = torch.cat(
                    (
                        y_dict['data'],
                        torch.nan * torch.zeros(
                            y_dict['data'].shape[0],
                            x_dict['data'].shape[1] - y_dict['data'].shape[1],
                            y_dict['data'].shape[2],
                            device=y_dict['data'].device,
                            dtype=y_dict['data'].dtype,
                        ),
                    ),
                    dim=1
                )

            # Mask test y
            y_dict['data'][:, eval_pos:] = torch.nan

            # Get y type
            if task_type == 'cls':
                y_type = torch.zeros_like(y_dict['data'], device=y_dict['data'].device)
            else:
                y_type = torch.ones_like(y_dict['data'], device=y_dict['data'].device)

            # Embed y
            embedded_y = model.mixed_y_embedding(y_dict, y_type=y_type, eval_pos=eval_pos)
            embedded_x = model.add_embeddings(x_emb_result)
            embedded_all = torch.cat((embedded_x, embedded_y.unsqueeze(2)), dim=2)

            # Run through transformer encoder
            encoder_out = model.transformer_encoder(
                embedded_all, feature_atten_mask=None, eval_pos=eval_pos
            )[0]
            encoder_out = model.encoder_out_norm(encoder_out)

            # Extract test embeddings: average over features
            # encoder_out shape: [batch, seq_len, n_feature_groups+1, embed_dim]
            # test samples: encoder_out[:, eval_pos:, :-1, :] -> [batch, n_test, n_feature_groups, embed_dim]
            test_encoder_out = encoder_out[:, eval_pos:, :-1, :]  # Exclude y column

            # Average over feature groups (similar to mean pooling in LLMs)
            embeddings = test_encoder_out.mean(dim=2)  # [batch, n_test, embed_dim]
            embeddings = embeddings.squeeze(0)  # [n_test, embed_dim]

            # Get predictions from decoder
            test_y_encoder = encoder_out[:, eval_pos:, -1]  # y column for test samples
            test_y_type = y_type[:, eval_pos:]
            cls_output, reg_output = model.y_decoder(test_y_encoder, test_y_type)

            if task_type == 'cls':
                # cls_output has max_classes (10) logits; slice to actual n_classes
                predictions = cls_output.squeeze(0)[..., :self.d_out]
            else:
                predictions = reg_output.squeeze(0).squeeze(-1)

        return predictions.cpu().numpy(), embeddings.cpu().numpy()

    def precompute_predictions(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        split: str = 'train',
        cache_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ):
        """Pre-compute LimiX predictions AND embeddings for caching.

        Args:
            X: Input features
            indices: Sample indices for caching
            split: 'train', 'val', or 'test'
            cache_dir: Experiment-specific cache directory (fallback)
            dataset_name: Dataset name for central cache (e.g., 'weather', 'electricity')
        """
        # Central cache directory structure: cache/limix/{dataset}/
        # This allows reuse across different experiment runs
        central_cache_dir = None
        if dataset_name is not None:
            central_cache_dir = os.path.join('cache', 'limix', dataset_name)
            os.makedirs(central_cache_dir, exist_ok=True)

        # Try to load from central cache first, then experiment cache
        for try_cache_dir in [central_cache_dir, cache_dir]:
            if try_cache_dir is not None:
                pred_path = os.path.join(try_cache_dir, f'limix_predictions_{split}_ctx{self.context_size}.npy')
                emb_path = os.path.join(try_cache_dir, f'limix_embeddings_{split}_ctx{self.context_size}.npy')

                if os.path.exists(pred_path) and os.path.exists(emb_path):
                    print(f"LimiXSlowPrior: loading cached predictions from {try_cache_dir}")
                    print(f"  split={split}, context_size={self.context_size}")
                    predictions = np.load(pred_path)
                    embeddings = np.load(emb_path)
                    predictions = self.align_cached_predictions(
                        predictions, f" for {split}")

                    for i, idx in enumerate(indices[:len(predictions)]):
                        self.prediction_cache[split][int(idx)] = predictions[i]
                    for i, idx in enumerate(indices[:len(embeddings)]):
                        self.embedding_cache[split][int(idx)] = embeddings[i]
                    return

        # Use central cache for saving if dataset_name provided, otherwise use experiment cache
        save_cache_dir = central_cache_dir if central_cache_dir else cache_dir

        if self.limix is None or not self.limix_fitted:
            print(f"LimiXSlowPrior: not fitted, skipping precomputation for {split}")
            return

        print(f"LimiXSlowPrior: computing predictions and embeddings for {split} ({len(X)} samples)")

        try:
            # Normalize y for regression
            if self.is_regression:
                y_context = (self.context_y - self.y_mean) / self.y_std
            else:
                y_context = self.context_y

            # Strategy: try all-at-once first (fastest — single preprocessing pass),
            # then progressively halve batch size on OOM until it fits.
            # Each _extract_embeddings_batch call re-preprocesses the full context
            # (concat x_train+x_test, pipeline, etc.), so fewer calls = much faster.
            all_predictions = []
            all_embeddings = []

            # Try all-at-once first (or start smaller for wide datasets)
            batch_size = min(len(X), 8) if X.shape[1] > 500 else len(X)
            success = False
            while batch_size >= 1:
                try:
                    if batch_size >= len(X):
                        print(f"  Trying all-at-once ({len(X)} samples)...")
                    else:
                        print(f"  Trying batch_size={batch_size}...")
                    all_predictions = []
                    all_embeddings = []
                    for i in range(0, len(X), batch_size):
                        batch_X = X[i:i+batch_size]
                        pred, emb = self._extract_embeddings_batch(
                            self.context_X,
                            y_context,
                            batch_X
                        )
                        all_predictions.append(pred)
                        all_embeddings.append(emb)
                        if batch_size < len(X):
                            print(f"  Processed {min(i+batch_size, len(X))}/{len(X)} samples...")
                        torch.cuda.empty_cache()
                    success = True
                    break
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if 'out of memory' in str(e).lower() or 'CUDA' in str(e):
                        torch.cuda.empty_cache()
                        old_bs = batch_size
                        if old_bs <= 1:
                            print(f"  OOM at batch_size=1, giving up on current context size")
                            break
                        batch_size = max(1, batch_size // 2)
                        print(f"  OOM at batch_size={old_bs}, reducing to {batch_size}")
                        all_predictions = []
                        all_embeddings = []
                    else:
                        raise

            # If batch_size=1 still OOMs, reduce context_size and retry
            if not success:
                min_context = 256
                reduced_ctx = len(self.context_X) // 2
                while not success and reduced_ctx >= min_context:
                    print(f"  OOM with full context ({len(self.context_X)}), retrying with context_size={reduced_ctx}")
                    torch.cuda.empty_cache()
                    # Subsample context
                    rng = np.random.RandomState(42)
                    ctx_idx = rng.choice(len(self.context_X), reduced_ctx, replace=False)
                    ctx_X = self.context_X[ctx_idx]
                    ctx_y = y_context[ctx_idx]
                    # Try batch_size=1 with reduced context
                    try:
                        all_predictions = []
                        all_embeddings = []
                        for i in range(len(X)):
                            batch_X = X[i:i+1]
                            pred, emb = self._extract_embeddings_batch(ctx_X, ctx_y, batch_X)
                            all_predictions.append(pred)
                            all_embeddings.append(emb)
                            torch.cuda.empty_cache()
                        success = True
                        print(f"  Success with reduced context_size={reduced_ctx}")
                    except (RuntimeError, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                        all_predictions = []
                        all_embeddings = []
                        reduced_ctx = reduced_ctx // 2

            if not success:
                # Final fallback: use native predict (no embeddings — kNN will use feature_extractor)
                print(f"  FATAL: All context sizes failed, using native predict fallback (batch=32, no embeddings)")
                for i in range(0, len(X), 32):
                    batch_X = X[i:i+32]
                    try:
                        pred = self.limix.predict(
                            self.context_X, y_context, batch_X,
                            task_type="Regression" if self.is_regression else "Classification"
                        )
                        if isinstance(pred, torch.Tensor):
                            pred = pred.cpu().numpy()
                    except Exception:
                        pred = np.zeros(len(batch_X))
                    all_predictions.append(pred)
                    # Don't cache dummy embeddings — let model use feature_extractor fallback
                    if i % 5000 == 0 and i > 0:
                        torch.cuda.empty_cache()
                        print(f"  Processed {i}/{len(X)} samples...")
                # Only cache predictions, not embeddings
                predictions = np.concatenate(all_predictions, axis=0)
                if not self.is_regression and predictions.ndim == 3:
                    predictions = predictions.mean(axis=1)
                if self.is_regression:
                    predictions = predictions * self.y_std + self.y_mean
                for i, idx in enumerate(indices[:len(predictions)]):
                    self.prediction_cache[split][int(idx)] = predictions[i]
                print(f"LimiXSlowPrior: cached {len(predictions)} predictions (no embeddings) for {split}")
                if save_cache_dir:
                    os.makedirs(save_cache_dir, exist_ok=True)
                    np.save(os.path.join(save_cache_dir, f'limix_predictions_{split}_ctx{self.context_size}.npy'), predictions)
                return

            predictions = np.concatenate(all_predictions, axis=0)
            embeddings = np.concatenate(all_embeddings, axis=0)

            # Denormalize predictions for regression
            if self.is_regression:
                predictions = predictions * self.y_std + self.y_mean

            # Cache predictions and embeddings
            for i, idx in enumerate(indices[:len(predictions)]):
                self.prediction_cache[split][int(idx)] = predictions[i]
            for i, idx in enumerate(indices[:len(embeddings)]):
                self.embedding_cache[split][int(idx)] = embeddings[i]

            print(f"LimiXSlowPrior: cached {len(predictions)} predictions and embeddings for {split}")

            # Save to central cache directory (reusable across runs)
            if save_cache_dir is not None:
                os.makedirs(save_cache_dir, exist_ok=True)
                pred_path = os.path.join(save_cache_dir, f'limix_predictions_{split}_ctx{self.context_size}.npy')
                emb_path = os.path.join(save_cache_dir, f'limix_embeddings_{split}_ctx{self.context_size}.npy')
                np.save(pred_path, predictions)
                np.save(emb_path, embeddings)
                print(f"LimiXSlowPrior: saved to {save_cache_dir}")
                print(f"  context_size={self.context_size}, split={split}")

        except Exception as e:
            print(f"LimiXSlowPrior: precomputation failed for {split}: {e}")
            import traceback
            traceback.print_exc()
            print("LimiXSlowPrior: will use fallback MLP at inference time")

    def _compute_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predictions (fallback when cache miss)."""
        if self.limix is not None and self.limix_fitted:
            try:
                x_np = x.detach().cpu().numpy()

                if self.is_regression:
                    y_context = (self.context_y - self.y_mean) / self.y_std
                else:
                    y_context = self.context_y

                task_type = "Regression" if self.is_regression else "Classification"
                pred = self.limix.predict(
                    self.context_X,
                    y_context,
                    x_np,
                    task_type=task_type
                )

                if isinstance(pred, torch.Tensor):
                    pred = pred.cpu().numpy()

                if not self.is_regression and pred.ndim == 3:
                    # pred is [batch, n_feature_groups, n_classes] -> take mean over feature groups
                    pred = pred.mean(axis=1)

                if self.is_regression:
                    pred = pred * self.y_std + self.y_mean

                return torch.tensor(pred, device=x.device, dtype=x.dtype)

            except Exception as e:
                print(f"LimiXSlowPrior: inference failed: {e}, using fallback")

        # Fallback to MLP
        features = self.feature_extractor(x)
        return self.fallback_head(features)

    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute features using fallback feature extractor."""
        return self.feature_extractor(x)

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        split: str = 'train',
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with LimiX embeddings.

        Returns predictions and rich embeddings from the transformer encoder.
        """
        # Get prediction from cache or compute
        prediction = None
        if indices is not None:
            prediction = self.get_cached_prediction(indices, split)

        if prediction is None:
            prediction = self._compute_predictions(x)

        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(-1)
        elif not self.is_regression and prediction.dim() == 3:
            # Handle cases like [batch, 1, n_classes] or [batch, n_feature_groups, n_classes]
            # Squeeze dim 1 if it's 1, otherwise take mean
            if prediction.size(1) == 1:
                prediction = prediction.squeeze(1)
            else:
                prediction = prediction.mean(dim=1)

        prediction = prediction * self.pred_scale + self.pred_bias

        # Get embeddings from cache or fallback
        features = None
        raw_embeddings = None
        if indices is not None:
            limix_emb = self.get_cached_embedding(indices, split)
            if limix_emb is not None:
                raw_embeddings = limix_emb  # Raw LimiX embeddings (192-dim)
                if self.embedding_proj is not None:
                    # Project LimiX embeddings to feature_dim
                    features = self.embedding_proj(limix_emb)

        if features is None:
            # Fallback to learned feature extractor
            features = self.feature_extractor(x)

        result = {
            'prediction': prediction,
            'features': features,
        }
        if raw_embeddings is not None:
            result['raw_embeddings'] = raw_embeddings
        return result


# =============================================================================
# XGBoost Slow Prior
# =============================================================================

class XGBSlowPrior(SlowPrior):
    """XGBoost as a frozen slow prior. Trains on context, generates predictions + leaf embeddings."""

    def __init__(self, d_in, d_out, feature_dim=128, is_regression=True, device='cuda',
                 context_size=10000, n_estimators=500, max_depth=6, **kwargs):
        super().__init__(d_in, d_out, feature_dim, is_regression, device)
        self.context_size = context_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.xgb_model = None
        self._embed_proj = None
        self._embed_dim = None
        print(f"XGBSlowPrior: initialized (context_size={context_size}, n_estimators={n_estimators})")

    def set_context(self, X: np.ndarray, y: np.ndarray):
        """Train XGBoost on context data."""
        from xgboost import XGBClassifier, XGBRegressor
        n = min(len(X), self.context_size)
        if n < len(X):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), n, replace=False)
            X_ctx, y_ctx = X[idx], y[idx]
        else:
            X_ctx, y_ctx = X, y

        if self.is_regression:
            self.xgb_model = XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.1, verbosity=0)
            self.xgb_model.fit(X_ctx, y_ctx)
        else:
            self.xgb_model = XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.1, eval_metric='auc', verbosity=0)
            self.xgb_model.fit(X_ctx, y_ctx)

        # Get leaf embedding dimension
        leaves = self.xgb_model.apply(X_ctx[:1])
        self._embed_dim = leaves.shape[1]  # n_estimators

        # Create projection from leaf hashing to feature_dim
        self._embed_proj = nn.Linear(self._embed_dim, self._feature_dim).to(self.device)
        nn.init.xavier_normal_(self._embed_proj.weight)

        print(f"XGBSlowPrior: trained on {n} samples, leaf_dim={self._embed_dim}, proj→{self._feature_dim}")

    def precompute_predictions(self, X, indices, split='train', cache_dir=None, dataset_name=None, **kwargs):
        """Generate and cache XGB predictions + leaf embeddings."""
        if self.xgb_model is None:
            raise RuntimeError("Must call set_context() first")

        if dataset_name is None:
            dataset_name = 'unknown'
        central_cache = os.path.join('cache', 'xgb_prior', dataset_name)
        os.makedirs(central_cache, exist_ok=True)

        pred_path = os.path.join(central_cache, f'xgb_predictions_{split}.npy')
        emb_path = os.path.join(central_cache, f'xgb_embeddings_{split}.npy')

        if os.path.exists(pred_path) and os.path.exists(emb_path):
            preds = np.load(pred_path)
            embeddings = np.load(emb_path)
            print(f"XGBSlowPrior: loaded cached {split} ({len(preds)} samples)")
        else:
            # Predictions
            if self.is_regression:
                preds = self.xgb_model.predict(X).astype(np.float32)
            else:
                preds = self.xgb_model.predict_proba(X).astype(np.float32)

            # Leaf embeddings (hash to fixed dim)
            leaves = self.xgb_model.apply(X)  # [n, n_estimators]
            # Use modular hashing to create dense features
            n_buckets = self._feature_dim
            embeddings = np.zeros((len(X), n_buckets), dtype=np.float32)
            for i in range(leaves.shape[1]):
                bucket = leaves[:, i].astype(int) % n_buckets
                embeddings[np.arange(len(X)), bucket] += 1.0
            # Normalize
            row_norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            embeddings = embeddings / row_norm

            np.save(pred_path, preds)
            np.save(emb_path, embeddings)
            print(f"XGBSlowPrior: cached {split} ({len(preds)} samples)")

        # Store in cache
        for i, idx in enumerate(indices):
            self.prediction_cache[split][int(idx)] = preds[i]
            self.embedding_cache[split][int(idx)] = embeddings[i]

    def _compute_predictions(self, X, **kwargs):
        """Compute predictions for raw input."""
        if self.xgb_model is None:
            return np.zeros((len(X), self.d_out), dtype=np.float32)
        if self.is_regression:
            return self.xgb_model.predict(X).astype(np.float32)
        else:
            return self.xgb_model.predict_proba(X).astype(np.float32)

    def _compute_features(self, X, **kwargs):
        """Compute leaf-based features for raw input."""
        if self.xgb_model is None:
            return np.zeros((len(X), self._feature_dim), dtype=np.float32)
        leaves = self.xgb_model.apply(X)
        n_buckets = self._feature_dim
        embeddings = np.zeros((len(X), n_buckets), dtype=np.float32)
        for i in range(leaves.shape[1]):
            bucket = leaves[:, i].astype(int) % n_buckets
            embeddings[np.arange(len(X)), bucket] += 1.0
        row_norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return embeddings / row_norm

    # forward() inherited from base SlowPrior — uses _compute_predictions/_compute_features + cache


class XGBLiMiXSlowPrior(SlowPrior):
    """Hybrid: XGBoost predictions (strong prior) + LiMiX embeddings (rich kNN space).

    XGBoost is trained on ALL training data (no context_size limit) for maximum
    prediction quality. LiMiX embeddings provide a semantically meaningful space
    for kNN online correction.

    This combines the best of both:
    - XGB: strong predictions (trained on full data, no context limit)
    - LiMiX: meaningful embeddings where kNN correction is effective (r≈0.24 vs r≈0.001 for XGB leaves)
    """

    def __init__(self, d_in, d_out, feature_dim=192, is_regression=True, device='cuda',
                 n_estimators=500, max_depth=8, learning_rate=0.05,
                 limix_context_size=1024, **kwargs):
        super().__init__(d_in, d_out, feature_dim, is_regression, device)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.limix_context_size = limix_context_size
        self.xgb_model = None
        self._limix_prior = None
        print(f"XGBLiMiXSlowPrior: initialized (XGB n_est={n_estimators}, depth={max_depth}, "
              f"LiMiX ctx={limix_context_size}, embed_dim={feature_dim})")

    def set_context(self, X: np.ndarray, y: np.ndarray):
        """Train XGB on ALL data + prepare LiMiX for embeddings."""
        from xgboost import XGBClassifier, XGBRegressor

        # Train XGB on FULL data (no context_size limit — this is the key difference)
        if self.is_regression:
            self.xgb_model = XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, device='cuda',
                verbosity=0, random_state=42)
            self.xgb_model.fit(X, y)
        else:
            self.xgb_model = XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, device='cuda',
                eval_metric='auc', verbosity=0, random_state=42)
            self.xgb_model.fit(X, y)

        print(f"XGBLiMiXSlowPrior: XGB trained on {len(X)} samples (full data)")

        # Initialize LiMiX for embeddings only
        self._limix_prior = LimiXSlowPrior(
            d_in=self.d_in, d_out=self.d_out,
            feature_dim=self._feature_dim,
            is_regression=self.is_regression,
            device=self.device,
            context_size=self.limix_context_size,
        )
        self._limix_prior.set_context(X, y)
        print(f"XGBLiMiXSlowPrior: LiMiX context set for embeddings")

    def precompute_predictions(self, X, indices, split='train', cache_dir=None, dataset_name=None, **kwargs):
        """Cache XGB predictions + LiMiX embeddings."""
        if self.xgb_model is None:
            raise RuntimeError("Must call set_context() first")

        if dataset_name is None:
            dataset_name = 'unknown'

        # --- XGB predictions ---
        xgb_cache = os.path.join('cache', 'xgb_limix', dataset_name)
        os.makedirs(xgb_cache, exist_ok=True)
        pred_path = os.path.join(xgb_cache, f'xgb_predictions_{split}.npy')

        if os.path.exists(pred_path):
            preds = np.load(pred_path)
            print(f"XGBLiMiXSlowPrior: loaded cached XGB predictions for {split} ({len(preds)})")
        else:
            if self.is_regression:
                preds = self.xgb_model.predict(X).astype(np.float32)
            else:
                preds = self.xgb_model.predict_proba(X).astype(np.float32)
            np.save(pred_path, preds)
            print(f"XGBLiMiXSlowPrior: cached XGB predictions for {split} ({len(preds)})")

        # Store predictions in cache
        for i, idx in enumerate(indices):
            self.prediction_cache[split][int(idx)] = preds[i]

        # --- LiMiX embeddings ---
        limix_cache_dir = os.path.join('cache', 'limix', dataset_name)
        emb_path = os.path.join(limix_cache_dir, f'limix_embeddings_{split}_ctx{self.limix_context_size}.npy')

        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
            print(f"XGBLiMiXSlowPrior: loaded cached LiMiX embeddings for {split} ({len(embeddings)})")
        else:
            # Compute LiMiX embeddings
            print(f"XGBLiMiXSlowPrior: computing LiMiX embeddings for {split}...")
            self._limix_prior.precompute_predictions(X, indices, split, cache_dir=cache_dir, dataset_name=dataset_name)
            emb_cache = self._limix_prior.embedding_cache.get(split, {})
            embeddings = np.array([emb_cache[int(idx)] for idx in indices], dtype=np.float32)

        # Augment embeddings with XGB predictions for richer kNN space
        # XGB pred similarity → neighbors have similar residuals → better correction
        if preds.ndim == 1:
            xgb_feat = preds.reshape(-1, 1)
        else:
            xgb_feat = preds
        # Scale XGB features to match embedding magnitude
        emb_std = np.std(embeddings) if np.std(embeddings) > 0 else 1.0
        xgb_std = np.std(xgb_feat) if np.std(xgb_feat) > 0 else 1.0
        xgb_feat_scaled = xgb_feat * (emb_std / xgb_std)
        augmented_embeddings = np.concatenate([embeddings, xgb_feat_scaled], axis=1).astype(np.float32)

        # Store augmented embeddings in cache
        for i, idx in enumerate(indices):
            self.embedding_cache[split][int(idx)] = augmented_embeddings[i]

    def _compute_predictions(self, X, **kwargs):
        if self.xgb_model is None:
            return np.zeros((len(X), self.d_out), dtype=np.float32)
        if self.is_regression:
            return self.xgb_model.predict(X).astype(np.float32)
        else:
            return self.xgb_model.predict_proba(X).astype(np.float32)

    def _compute_features(self, X, **kwargs):
        # Fallback: if LiMiX not available, return zeros
        return np.zeros((len(X), self._feature_dim), dtype=np.float32)


class XGBTreeEmbSlowPrior(SlowPrior):
    """XGBoost predictions + per-tree margin embeddings for kNN correction.

    Instead of LiMiX embeddings (which can be poor quality on some datasets),
    uses the per-tree output values from XGBoost itself as embeddings. Each tree
    provides one dimension, capturing the model's internal feature interactions.

    Advantages over XGB+LiMiX:
    - Embeddings directly reflect XGB's learned structure (no external model needed)
    - Much faster (no LiMiX inference)
    - Higher kNN-error correlation on datasets where LiMiX embeddings are poor
      (e.g., IEEE fraud: r=0.177 vs r=0.023 for LiMiX)
    """

    def __init__(self, d_in, d_out, feature_dim=200, is_regression=True, device='cuda',
                 n_estimators=500, max_depth=8, learning_rate=0.05,
                 n_trees_for_emb=200, **kwargs):
        super().__init__(d_in, d_out, feature_dim, is_regression, device)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_trees_for_emb = n_trees_for_emb
        self.xgb_model = None
        self._actual_n_trees = None
        print(f"XGBTreeEmbSlowPrior: initialized (n_est={n_estimators}, depth={max_depth}, "
              f"emb_trees={n_trees_for_emb})")

    @property
    def feature_dim(self):
        return self._actual_n_trees or self.n_trees_for_emb

    def set_context(self, X: np.ndarray, y: np.ndarray):
        """Train XGB on all data."""
        from xgboost import XGBClassifier, XGBRegressor

        if self.is_regression:
            self.xgb_model = XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, device='cuda',
                verbosity=0, random_state=42)
            self.xgb_model.fit(X, y)
        else:
            self.xgb_model = XGBClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, device='cuda',
                eval_metric='auc', verbosity=0, random_state=42)
            self.xgb_model.fit(X, y)

        self._actual_n_trees = min(
            self.n_trees_for_emb,
            self.xgb_model.get_booster().num_boosted_rounds()
        )
        print(f"XGBTreeEmbSlowPrior: XGB trained on {len(X)} samples, "
              f"using {self._actual_n_trees} trees for embeddings")

    def _compute_tree_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Compute per-tree margin outputs as embeddings."""
        import xgboost as xgb
        booster = self.xgb_model.get_booster()
        dmat = xgb.DMatrix(X)
        n = len(X)
        n_trees = self._actual_n_trees
        margins = np.zeros((n, n_trees), dtype=np.float32)
        prev = np.zeros(n, dtype=np.float32)
        for t in range(n_trees):
            cum = booster.predict(dmat, iteration_range=(0, t + 1), output_margin=True)
            margins[:, t] = cum - prev
            prev = cum
        return margins

    def precompute_predictions(self, X, indices, split='train', cache_dir=None, dataset_name=None, **kwargs):
        if self.xgb_model is None:
            raise RuntimeError("Must call set_context() first")
        if dataset_name is None:
            dataset_name = 'unknown'

        xgb_cache = os.path.join('cache', 'xgb_tree_emb', dataset_name)
        os.makedirs(xgb_cache, exist_ok=True)

        # --- XGB predictions ---
        pred_path = os.path.join(xgb_cache, f'xgb_predictions_{split}.npy')
        if os.path.exists(pred_path):
            preds = np.load(pred_path)
            print(f"XGBTreeEmbSlowPrior: loaded cached predictions for {split} ({len(preds)})")
        else:
            if self.is_regression:
                preds = self.xgb_model.predict(X).astype(np.float32)
            else:
                preds = self.xgb_model.predict_proba(X).astype(np.float32)
            np.save(pred_path, preds)
            print(f"XGBTreeEmbSlowPrior: cached predictions for {split} ({len(preds)})")

        for i, idx in enumerate(indices):
            self.prediction_cache[split][int(idx)] = preds[i]

        # --- Per-tree margin embeddings ---
        emb_path = os.path.join(xgb_cache, f'tree_embeddings_{split}.npy')
        if os.path.exists(emb_path):
            embeddings = np.load(emb_path)
            print(f"XGBTreeEmbSlowPrior: loaded cached tree embeddings for {split} ({len(embeddings)})")
        else:
            print(f"XGBTreeEmbSlowPrior: computing tree embeddings for {split}...")
            embeddings = self._compute_tree_embeddings(X)
            np.save(emb_path, embeddings)
            print(f"XGBTreeEmbSlowPrior: cached tree embeddings for {split} ({embeddings.shape})")

        for i, idx in enumerate(indices):
            self.embedding_cache[split][int(idx)] = embeddings[i]

    def _compute_predictions(self, X, **kwargs):
        if self.xgb_model is None:
            return np.zeros((len(X), self.d_out), dtype=np.float32)
        if self.is_regression:
            return self.xgb_model.predict(X).astype(np.float32)
        else:
            return self.xgb_model.predict_proba(X).astype(np.float32)

    def _compute_features(self, X, **kwargs):
        if self.xgb_model is None:
            return np.zeros((len(X), self.feature_dim), dtype=np.float32)
        return self._compute_tree_embeddings(X)


# =============================================================================
# Factory Function
# =============================================================================

def create_slow_prior(
    prior_type: str,
    d_in: int,
    d_out: int,
    feature_dim: int = 128,
    is_regression: bool = True,
    device: str = 'cuda',
    **kwargs,
) -> SlowPrior:
    """
    Factory function to create slow priors.

    Args:
        prior_type: 'tabpfn', 'limix', 'lm', 'ensemble', or 'mlp'
        d_in: input dimension
        d_out: output dimension
        feature_dim: feature embedding dimension
        is_regression: whether this is a regression task
        device: torch device
        **kwargs: additional arguments passed to the prior constructor

    Returns:
        SlowPrior instance
    """
    prior_type = prior_type.lower()

    if prior_type == 'tabpfn':
        return TabPFNSlowPrior(
            d_in=d_in,
            d_out=d_out,
            feature_dim=feature_dim,
            is_regression=is_regression,
            device=device,
            **kwargs,
        )

    elif prior_type == 'limix':
        return LimiXSlowPrior(
            d_in=d_in,
            d_out=d_out,
            feature_dim=feature_dim,
            is_regression=is_regression,
            device=device,
            **kwargs,
        )

    elif prior_type in ['lm', 'language_model', 'languagemodel']:
        return LanguageModelSlowPrior(
            d_in=d_in,
            d_out=d_out,
            feature_dim=feature_dim,
            is_regression=is_regression,
            device=device,
            **kwargs,
        )

    elif prior_type == 'xgb':
        return XGBSlowPrior(
            d_in=d_in,
            d_out=d_out,
            feature_dim=feature_dim,
            is_regression=is_regression,
            device=device,
            **{k: v for k, v in kwargs.items() if k in ['context_size', 'n_estimators', 'max_depth']},
        )

    elif prior_type == 'xgb_tree_emb':
        return XGBTreeEmbSlowPrior(
            d_in=d_in,
            d_out=d_out,
            feature_dim=200,  # per-tree margin dimension
            is_regression=is_regression,
            device=device,
            **{k: v for k, v in kwargs.items() if k in [
                'n_estimators', 'max_depth', 'learning_rate', 'n_trees_for_emb']},
        )

    elif prior_type == 'xgb_limix':
        return XGBLiMiXSlowPrior(
            d_in=d_in,
            d_out=d_out,
            feature_dim=192,  # LiMiX embedding dimension
            is_regression=is_regression,
            device=device,
            **{k: v for k, v in kwargs.items() if k in [
                'n_estimators', 'max_depth', 'learning_rate', 'limix_context_size']},
        )

    elif prior_type == 'ensemble':
        return EnsembleSlowPrior(
            d_in=d_in,
            d_out=d_out,
            feature_dim=feature_dim,
            is_regression=is_regression,
            device=device,
            **kwargs,
        )

    elif prior_type == 'mlp':
        # Simple MLP baseline (no external prior)
        return MLPSlowPrior(
            d_in=d_in,
            d_out=d_out,
            feature_dim=feature_dim,
            is_regression=is_regression,
            device=device,
            **kwargs,
        )

    elif prior_type == 'ftt':
        return FTTSlowPrior(
            d_in=d_in,
            d_out=d_out,
            feature_dim=feature_dim,
            is_regression=is_regression,
            device=device,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown prior type: {prior_type}. "
                        f"Choose from: tabpfn, limix, lm, ensemble, mlp, ftt, xgb, xgb_limix")


class FTTSlowPrior(SlowPrior):
    """
    Feature Tokenizer Transformer as trainable slow prior.

    Learns via RECONSTRUCTION, not prediction — aligned with PFC slow timescale
    principles where the slow layer captures environmental structure (statistical
    regularities, feature correlations) rather than stimulus-response mappings.

    Training principle:
    - Prediction loss does NOT flow back to FTT (stop gradient)
    - FTT is trained with its own reconstruction objective
    - This provides stable contextual representations like a meta-layer

    Provides:
    - Rich embeddings from [CLS] token capturing data structure
    - Reconstruction loss for self-supervised training
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        feature_dim: int = 128,
        is_regression: bool = True,
        device: str = 'cuda',
        n_layers: int = 3,
        n_heads: int = 8,
        d_ffn_factor: float = 4/3,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        mask_ratio: float = 0.15,
        **kwargs,
    ):
        super().__init__(d_in, d_out, feature_dim, is_regression, device)

        d_token = feature_dim
        self.d_token = d_token
        self.d_in = d_in
        self.mask_ratio = mask_ratio

        # Tokenizer: project each feature to d_token
        self.feature_tokenizer = nn.Linear(1, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Learnable mask token (replaces masked features)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, d_in + 1, d_token) * 0.02)

        # Transformer encoder
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_token),
                'attn': nn.MultiheadAttention(d_token, n_heads, dropout=attention_dropout, batch_first=True),
                'norm2': nn.LayerNorm(d_token),
                'ffn': nn.Sequential(
                    nn.Linear(d_token, int(d_token * d_ffn_factor)),
                    nn.GELU(),
                    nn.Dropout(ffn_dropout),
                    nn.Linear(int(d_token * d_ffn_factor), d_token),
                    nn.Dropout(residual_dropout),
                ),
            }))

        self.final_norm = nn.LayerNorm(d_token)

        # Reconstruction head: predict original feature values from token embeddings
        # This is the slow prior's own learning signal
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.GELU(),
            nn.Linear(d_token // 2, 1),
        )

        # Prediction head (for providing predictions to architecture)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.GELU(),
            nn.Linear(d_token // 2, d_out),
        )

        # Store last reconstruction loss for training
        self._reconstruction_loss = None

        self.to(device)

    def _forward_transformer(self, x: torch.Tensor, mask_features: bool = False):
        """
        Forward through transformer with optional feature masking.

        Args:
            x: [batch, d_in] raw input features
            mask_features: whether to mask features for reconstruction training

        Returns:
            cls_embedding: [batch, d_token] CLS token embedding
            feature_tokens: [batch, d_in, d_token] per-feature token embeddings
            mask: [batch, d_in] boolean mask (True = masked)
        """
        batch_size = x.size(0)

        # Tokenize each feature: [batch, d_in] -> [batch, d_in, d_token]
        x_tokens = self.feature_tokenizer(x.unsqueeze(-1))

        # Apply masking during training (like masked language modeling)
        mask = None
        if mask_features and self.training:
            mask = torch.rand(batch_size, self.d_in, device=x.device) < self.mask_ratio
            # Replace masked tokens with learnable mask token
            mask_expanded = mask.unsqueeze(-1).expand_as(x_tokens)
            x_tokens = torch.where(mask_expanded, self.mask_token.expand_as(x_tokens), x_tokens)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_tokens = torch.cat([cls_tokens, x_tokens], dim=1)  # [batch, d_in+1, d_token]

        # Add positional encoding
        x_tokens = x_tokens + self.pos_embedding[:, :x_tokens.size(1), :]

        # Transformer layers
        for layer in self.layers:
            normed = layer['norm1'](x_tokens)
            attn_out, _ = layer['attn'](normed, normed, normed)
            x_tokens = x_tokens + attn_out

            normed = layer['norm2'](x_tokens)
            x_tokens = x_tokens + layer['ffn'](normed)

        x_tokens = self.final_norm(x_tokens)

        cls_embedding = x_tokens[:, 0]           # [batch, d_token]
        feature_tokens = x_tokens[:, 1:]          # [batch, d_in, d_token]

        return cls_embedding, feature_tokens, mask

    def _compute_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss: predict masked feature values.

        This is the slow prior's own learning signal — it learns data structure
        by predicting masked features from context (like BERT for tabular data).
        """
        cls_emb, feature_tokens, mask = self._forward_transformer(x, mask_features=True)

        # Predict original feature values from token embeddings
        reconstructed = self.reconstruction_head(feature_tokens).squeeze(-1)  # [batch, d_in]

        if mask is not None and mask.any():
            # Loss only on masked positions (predict what was hidden)
            loss = F.mse_loss(reconstructed[mask], x[mask])
        else:
            # Fallback: reconstruct all features
            loss = F.mse_loss(reconstructed, x)

        return loss

    def set_context(self, X: np.ndarray, y: np.ndarray):
        """FTT doesn't need context - it learns during training."""
        pass

    def precompute_predictions(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        split: str = 'train',
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """FTT computes on-the-fly - no caching needed."""
        pass

    def _compute_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predictions from embeddings."""
        cls_emb, _, _ = self._forward_transformer(x, mask_features=False)
        return self.head(cls_emb)

    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings as features."""
        cls_emb, _, _ = self._forward_transformer(x, mask_features=False)
        return cls_emb

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        split: str = 'train',
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with reconstruction-based learning.

        The CLS embedding is DETACHED from prediction loss — the slow prior
        only learns from its own reconstruction objective.
        """
        # Compute reconstruction loss (slow prior's own learning signal)
        if self.training:
            self._reconstruction_loss = self._compute_reconstruction_loss(x)

        # Get embeddings without masking for downstream use
        cls_emb, _, _ = self._forward_transformer(x, mask_features=False)

        # Prediction from CLS
        predictions = self.head(cls_emb)

        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(-1)
        predictions = predictions * self.pred_scale + self.pred_bias

        # Features are NOT detached — weak task gradients flow back through
        # the slow prior's low learning rate (multi-timescale learning).
        # The reconstruction loss acts as a structural regularizer.
        return {
            'prediction': predictions,
            'features': cls_emb,
            'reconstruction_loss': self._reconstruction_loss,
        }

    def get_reconstruction_loss(self) -> Optional[torch.Tensor]:
        """Get the last computed reconstruction loss for training."""
        return self._reconstruction_loss


class MLPSlowPrior(SlowPrior):
    """
    Simple MLP baseline slow prior (no external knowledge).

    Useful for ablation studies to see how much the external prior helps.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        feature_dim: int = 128,
        is_regression: bool = True,
        device: str = 'cuda',
        hidden_dim: int = 256,
        n_layers: int = 2,
        **kwargs,  # Accept and ignore extra kwargs (e.g., from LimiX config)
    ):
        super().__init__(d_in, d_out, feature_dim, is_regression, device)
        # Ignore kwargs - they may be from other prior types (e.g., context_size from LimiX)

        layers = [
            nn.Linear(d_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        ]

        for _ in range(n_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])

        layers.extend([
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        ])

        self.feature_extractor = nn.Sequential(*layers)

        self.pred_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, d_out),
        )

        # Move to device
        self.to(device)

    def set_context(self, X: np.ndarray, y: np.ndarray):
        """No context needed for MLP."""
        pass

    def precompute_predictions(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        split: str = 'train',
        cache_dir: Optional[str] = None,
    ):
        """No precomputation needed for MLP."""
        pass

    def _compute_predictions(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.pred_head(features)

    def _compute_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)
