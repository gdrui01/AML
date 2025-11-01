# feature_selectors.py
import numpy as np
from typing import Optional, Dict, Any

from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, TransformerMixin



class BaseSelector:
    """Minimal, sklearn-like interface used by main.py"""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X
    def get_support(self):
        # For true feature selection (subset) this returns a boolean mask.
        # For PCA or 'none' it returns None (no subset of original features).
        return None


class KBestSelector(BaseSelector):
    def __init__(self, k: int = 300):
        self.k = k
        self.selector = SelectKBest(score_func=f_regression, k=self.k)

    def fit(self, X, y=None):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def get_support(self):
        return self.selector.get_support()


class LassoSelector(BaseSelector):
    def __init__(self, cv: int = 5, random_state: int = 0, max_iter: int = 500):
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        # ✅ allow more iterations + parallel jobs
        self.model = LassoCV(
            cv=self.cv,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_jobs=-1
        )
        self.selector = None

    def fit(self, X, y=None):
        self.model.fit(X, y)
        self.selector = SelectFromModel(self.model, prefit=True)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def get_support(self):
        return self.selector.get_support()

class RidgeSelector(BaseSelector):
    def __init__(self, alphas, cv: int = 5):
        self.alphas = alphas
        self.cv = cv
        self.model = RidgeCV(alphas=self.alphas, cv=self.cv)
        self.selector = None

    def fit(self, X, y=None):
        self.model.fit(X, y)
        self.selector = SelectFromModel(self.model, prefit=True)
        return self

    def transform(self, X):
        return self.selector.transform(X)

    def get_support(self):
        return self.selector.get_support()


class PCASelector(BaseSelector):
    """Not a true 'feature selector': reduces to components (no boolean mask)."""
    def __init__(self, explained_variance_threshold: float = 0.95, random_state: int = 0):
        self.threshold = explained_variance_threshold
        self.random_state = random_state
        self.pca = None
        self.n_components_ = None

    def fit(self, X, y=None):
        # probe to decide number of components
        pca_probe = PCA(n_components=min(X.shape[1], X.shape[0]), random_state=self.random_state, svd_solver='full')
        pca_probe.fit(X)
        cum = np.cumsum(pca_probe.explained_variance_ratio_)
        n = int(np.searchsorted(cum, self.threshold) + 1)
        n = min(n, X.shape[1])
        self.pca = PCA(n_components=n, random_state=self.random_state, svd_solver='full').fit(X)
        self.n_components_ = n
        return self

    def transform(self, X):
        return self.pca.transform(X)

    # get_support stays None (components are linear combos, not a subset)


def make_feature_selector(
    method: str,
    random_state: int = 0,
    **kwargs: Dict[str, Any]
) -> Optional[BaseSelector]:
    """
    method in {'none','kbest','lasso','ridge','pca'}
    kwargs:
      - k (for kbest)
      - cv (for lasso/ridge)
      - alphas (for ridge)
      - explained_variance_threshold (for pca)
    """
    method = (method or "none").lower()

    if method == "none":
        return BaseSelector()

    if method == "kbest":
        k = kwargs.get("k", 300)
        return KBestSelector(k=k)

    if method == "lasso":
        cv = kwargs.get("cv", 5)
        return LassoSelector(cv=cv, random_state=random_state)

    if method == "ridge":
        alphas = kwargs.get("alphas", np.logspace(-4, 4, 41))
        cv = kwargs.get("cv", 5)
        return RidgeSelector(alphas=alphas, cv=cv)

    if method == "pca":
        thr = kwargs.get("explained_variance_threshold", 0.95)
        return PCASelector(explained_variance_threshold=thr, random_state=random_state)

    if method == "autoencoder":
        return AutoencoderSelector(
            encoding_dim=kwargs.get("encoding_dim", 64),
            max_iter=kwargs.get("max_iter", 500),
            random_state=random_state
        )
    # default
    return BaseSelector()

class AutoencoderSelector(BaseEstimator, TransformerMixin):
    """
    Lightweight 'autoencoder-like' feature extractor using scikit-learn MLPRegressor.
    It learns to reconstruct inputs and uses the hidden layer as compressed features.
    """
    def __init__(self, encoding_dim=64, max_iter=500, random_state=0):
        self.encoding_dim = encoding_dim
        self.max_iter = max_iter
        self.random_state = random_state
        self.encoder = None

    def fit(self, X, y=None):
        input_dim = X.shape[1]
        # Train MLP to reconstruct X (unsupervised)
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.encoding_dim,),
            activation='relu',
            solver='adam',
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model.fit(X, X)
        # Extract learned weights for transformation
        self.encoder_weights_ = self.model.coefs_[0]
        self.encoder_bias_ = self.model.intercepts_[0]
        return self

    def transform(self, X):
        # Apply first layer transformation manually (hidden layer activations)
        hidden = np.dot(X, self.encoder_weights_) + self.encoder_bias_
        return np.maximum(hidden, 0)  # ReLU

    def get_support(self):
        return None  # not a subset selector, it's an embedding
