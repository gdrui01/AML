import os
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from feature_selectors import make_feature_selector
from regressors import make_regressor
from outlier_detectors import make_outlier_detector

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence and benign warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_NUMBER = 696969

# ======= Feature selection defaults =======
FEATURE_SELECTION_METHOD = "kbest"
FS_PARAMS = {
    "k": 500,   # increased: allow more features to help deep models
    "cv": 10,   # more robust internal CV for feature selection
    "alphas": np.logspace(-5, 5, 81),
    "explained_variance_threshold": 0.99,
}

# ======= Regression methods (tuned for better convergence) =======
REGRESSORS = [
    ("ridge", {"cv": 10, "alphas": np.logspace(-5, 5, 81)}),
    ("elasticnet", {"cv": 10, "l1_ratio": [0.1, 0.5, 0.9]}),
    ("huber", {"alpha": 1e-5, "epsilon": 1.5, "max_iter": 5000}),
    ("svr", {"C": 50.0, "gamma": "scale"}),  # more flexible margin
    ("rf", {"n_estimators": 1000, "max_depth": None, "n_jobs": -1}),
    ("hgbt", {"max_iter": 1500, "learning_rate": 0.03}),
    ("mlp", {
        "hidden_layer_sizes": (256, 128, 64),
        "learning_rate_init": 0.0005,
        "max_iter": 3000,
        "early_stopping": False,
        "n_iter_no_change": 50
    }),
]

# ======= Outlier detection methods =======
OUTLIER_METHODS = [
    ("none", {}),
    ("isoforest", {"n_estimators": 500, "contamination": 0.02, "random_state": RANDOM_NUMBER}),
    ("lof", {"contamination": 0.02, "n_neighbors": 30}),
    ("svm", {"nu": 0.02, "gamma": "scale"}),
    ("elliptic", {"contamination": 0.02}),
]


# ======= Helper functions =======
def detect_outliers_mask(X, method="isoforest", od_params=None):
    detector = make_outlier_detector(method, **(od_params or {}))
    if detector is None:
        return np.ones(X.shape[0], dtype=bool)
    detector.fit(X)
    preds = detector.fit_predict(X) if hasattr(detector, "fit_predict") else detector.predict(X)
    return preds == 1


def process_missing_vals(X, imputer=None):
    if imputer is None:
        imputer = SimpleImputer(strategy="mean")
        X_imp = imputer.fit_transform(X)
        return X_imp, imputer
    else:
        X_imp = imputer.transform(X)
        return X_imp, imputer


def build_model(reg_name="ridge", reg_params=None):
    return make_regressor(reg_name, **(reg_params or {}))


def output_submission(test_ids, predictions, suffix=""):
    fname = f"submission{('_' + suffix) if suffix else ''}.csv"
    pd.DataFrame({'id': test_ids, 'y': predictions}).to_csv(fname, index=False)
    print(f"submission written to file: {fname}")


def apply_feature_selection(X_tr_clean, y_tr_clean, X_va_imp_sc, X_te_imp_sc=None,
                            method=None, fs_params=None):
    method = (method or FEATURE_SELECTION_METHOD)
    params = dict(FS_PARAMS)
    if fs_params:
        params.update(fs_params)

    selector = make_feature_selector(method, random_state=RANDOM_NUMBER, **params)
    selector.fit(X_tr_clean, y_tr_clean)
    X_tr_sel = selector.transform(X_tr_clean)
    X_va_sel = selector.transform(X_va_imp_sc)
    X_te_sel = selector.transform(X_te_imp_sc) if X_te_imp_sc is not None else None
    mask = selector.get_support()
    return X_tr_sel, X_va_sel, X_te_sel, mask


# ======= Cross-validation with repeats =======
def cross_validate_model(X, y, n_splits=10, n_repeats=3, method=None, fs_params=None,
                         reg_name="ridge", reg_params=None,
                         od_method="isoforest", od_params=None):

    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_NUMBER)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(rkf.split(X), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Impute
        X_tr_imp, imp = process_missing_vals(X_tr, imputer=None)
        X_va_imp, _ = process_missing_vals(X_va, imputer=imp)

        # Pre-scale
        pre_scaler = StandardScaler()
        X_tr_imp_sc = pre_scaler.fit_transform(X_tr_imp)
        X_va_imp_sc = pre_scaler.transform(X_va_imp)

        # Outlier detection
        inliers = detect_outliers_mask(X_tr_imp_sc, method=od_method, od_params=od_params)
        X_tr_clean = X_tr_imp_sc[inliers]
        y_tr_clean = y_tr.to_numpy()[inliers]

        # Feature selection
        X_tr_sel, X_va_sel, _, _ = apply_feature_selection(
            X_tr_clean, y_tr_clean, X_va_imp_sc,
            method=method, fs_params=fs_params
        )

        # Post-scale
        post_scaler = StandardScaler()
        X_tr_final = post_scaler.fit_transform(X_tr_sel)
        X_va_final = post_scaler.transform(X_va_sel)

        # Train model (with larger iterations)
        model = build_model(reg_name, reg_params)
        model.fit(X_tr_final, y_tr_clean)

        # Evaluate
        y_va_pred = model.predict(X_va_final)
        score = r2_score(y_va, y_va_pred)
        fold_scores.append(score)
        print(f"Fold {fold}: R^2 = {score:.5f}")

    mean, std = float(np.mean(fold_scores)), float(np.std(fold_scores))
    print(f"Repeated CV mean R^2 = {mean:.5f} ± {std:.5f}")
    return np.array(fold_scores)


def run_full_pipeline_for_method(method_name, fs_params,
                                 reg_name, reg_params,
                                 X_train_feat, y_train_trgt, X_test_feat, test_ids,
                                 od_method="isoforest", od_params=None):

    print("\n" + "="*100)
    print(f"Running pipeline: Outlier={od_method} | Selector={method_name} | Regressor={reg_name}")
    print("="*100)

    scores = cross_validate_model(X_train_feat, y_train_trgt, n_splits=10, n_repeats=3,
                                  method=method_name, fs_params=fs_params,
                                  reg_name=reg_name, reg_params=reg_params,
                                  od_method=od_method, od_params=od_params)
    mean, std = float(np.mean(scores)), float(np.std(scores))
    print(f"[{od_method} + {method_name} + {reg_name}] CV mean R^2 = {mean:.5f} ± {std:.5f}")

    # Final train on all data
    X_full_imp, imp = process_missing_vals(X_train_feat, imputer=None)
    X_test_imp, _ = process_missing_vals(X_test_feat, imputer=imp)
    pre_scaler = StandardScaler()
    X_full_sc = pre_scaler.fit_transform(X_full_imp)
    X_test_sc = pre_scaler.transform(X_test_imp)

    inliers_full = detect_outliers_mask(X_full_sc, method=od_method, od_params=od_params)
    X_full_clean = X_full_sc[inliers_full]
    y_full_clean = y_train_trgt.to_numpy()[inliers_full]

    X_full_sel, _, X_test_sel, selected_mask = apply_feature_selection(
        X_full_clean, y_full_clean, X_test_sc, X_te_imp_sc=X_test_sc,
        method=method_name, fs_params=fs_params
    )

    post_scaler = StandardScaler()
    X_full_final = post_scaler.fit_transform(X_full_sel)
    X_test_final = post_scaler.transform(X_test_sel)

    model = build_model(reg_name, reg_params)
    model.fit(X_full_final, y_full_clean)
    y_test_pred = model.predict(X_test_final)

    output_submission(test_ids, y_test_pred, suffix=f"{od_method}_{method_name}_{reg_name}")
    return mean, std


def main():
    base = "eth-aml-2025-project-1"
    X_train = pd.read_csv(os.path.join(base, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(base, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(base, "X_test.csv"))

    test_ids = X_test['id'].copy()
    X_train_feat = X_train.drop(columns=['id'])
    y_train_trgt = y_train['y']
    X_test_feat = X_test.drop(columns=['id'])

    methods = [
        ("none", {}),
        ("kbest", {"k": 500}),
        ("lasso", {"cv": 10}),
        ("ridge", {"cv": 10, "alphas": np.logspace(-5, 5, 81)}),
        ("pca", {"explained_variance_threshold": 0.99}),
        ("autoencoder", {"encoding_dim": 128, "max_iter": 1000}),
    ]

    results = []
    for od_method, od_params in OUTLIER_METHODS:
        for method_name, fs_params in methods:
            for reg_name, reg_params in REGRESSORS:
                mean, std = run_full_pipeline_for_method(
                    method_name, fs_params, reg_name, reg_params,
                    X_train_feat, y_train_trgt, X_test_feat, test_ids,
                    od_method=od_method, od_params=od_params
                )
                results.append((od_method, method_name, reg_name, mean, std))

    print("\nSummary (Repeated CV mean R^2 ± std):")
    for o, m, r, mean, std in results:
        print(f"- {o:>10} | {m:>10} + {r:<10}: {mean:.5f} ± {std:.5f}")


if __name__ == "__main__":
    main()
