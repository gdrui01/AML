import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor  # ✅ Added

def make_regressor(method: str, **kwargs):
    """
    Returns a regression model by name.

    method in {'ridge','lasso','elasticnet','huber','svr','knn','rf','hgbt','mlp'}
    """
    m = (method or "ridge").lower()

    if m == "ridge":
        alphas = kwargs.get("alphas", np.logspace(-4, 4, 41))
        cv = kwargs.get("cv", 5)
        return RidgeCV(alphas=alphas, cv=cv)

    if m == "lasso":
        cv = kwargs.get("cv", 5)
        return LassoCV(
            cv=cv,
            random_state=kwargs.get("random_state", 0),
            max_iter=50,
            n_jobs=-1
        )

    if m == "elasticnet":
        cv = kwargs.get("cv", 5)
        l1_ratio = kwargs.get("l1_ratio", [0.1, 0.5, 0.9])
        return ElasticNetCV(
            l1_ratio=l1_ratio,
            alphas=np.logspace(-4, 2, 30),
            cv=cv,
            random_state=kwargs.get("random_state", 0),
            max_iter=50,
            n_jobs=-1
        )

    if m == "huber":
        return HuberRegressor(
            alpha=kwargs.get("alpha", 1e-4),
            epsilon=kwargs.get("epsilon", 1.35),
            max_iter=1000
        )

    if m == "svr":
        return SVR(
            kernel="rbf",
            C=kwargs.get("C", 10.0),
            gamma=kwargs.get("gamma", "scale")
        )

    if m == "knn":
        return KNeighborsRegressor(
            n_neighbors=kwargs.get("n_neighbors", 15),
            weights=kwargs.get("weights", "distance")
        )

    if m == "rf":
        return RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 400),
            n_jobs=-1,
            random_state=kwargs.get("random_state", 0)
        )

    if m == "hgbt":
        return HistGradientBoostingRegressor(
            max_iter=kwargs.get("max_iter", 10),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=kwargs.get("random_state", 0)
        )

    # ✅ NEW: MLPRegressor (neural network)
    if m == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (128, 64)),
            activation=kwargs.get("activation", "relu"),
            solver=kwargs.get("solver", "adam"),
            learning_rate_init=kwargs.get("learning_rate_init", 0.001),
            alpha=kwargs.get("alpha", 1e-4),
            max_iter=kwargs.get("max_iter", 1000),
            early_stopping=True,
            random_state=kwargs.get("random_state", 0)
        )

    # Default fallback
    return RidgeCV(alphas=np.logspace(-4, 4, 41), cv=5)
