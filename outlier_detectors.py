# outlier_detectors.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM

def make_outlier_detector(method="isoforest", **kwargs):
    """
    Returns an outlier detector by name.
    method in {'none', 'isoforest', 'lof', 'svm', 'elliptic'}
    """
    method = (method or "none").lower()

    if method == "none":
        return None

    if method == "isoforest":
        return IsolationForest(
            n_estimators=kwargs.get("n_estimators", 150),
            max_samples=kwargs.get("max_samples", "auto"),
            contamination=kwargs.get("contamination", 0.03),
            random_state=kwargs.get("random_state", 0)
        )

    if method == "lof":
        return LocalOutlierFactor(
            n_neighbors=kwargs.get("n_neighbors", 20),
            contamination=kwargs.get("contamination", 0.03),
            novelty=True,  # must be True to use .fit_predict on unseen data
            n_jobs=-1
        )

    if method == "svm":
        return OneClassSVM(
            kernel=kwargs.get("kernel", "rbf"),
            gamma=kwargs.get("gamma", "scale"),
            nu=kwargs.get("nu", 0.03)
        )

    if method == "elliptic":
        return EllipticEnvelope(
            contamination=kwargs.get("contamination", 0.03),
            random_state=kwargs.get("random_state", 0)
        )

    raise ValueError(f"Unknown outlier detector: {method}")
