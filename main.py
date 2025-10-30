import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import numpy as np

RANDOM_NUMBER = 696969

def detect_outliers(X, y):
    iso_forest = IsolationForest(n_estimators=150, max_samples='auto', contamination=0.069, random_state=RANDOM_NUMBER)
    prediction = iso_forest.fit_predict(X)
    inliners = prediction == 1
    return X.loc[inliners], y.loc[inliners]

def select_features(X, y, explained_variance_threshold=0.95, max_features=None, random_state=RANDOM_NUMBER):
    if max_features is None:
        max_features = X.shape[1]

    pca_probe = PCA(n_components=max_features, random_state=random_state)
    pca_probe.fit(X)
    cumulative_var = np.cumsum(pca_probe.explained_variance_ratio_)
    n_features = np.searchsorted(cumulative_var, explained_variance_threshold) + 1
    n_features = min(n_features, max_features)

    pca = PCA(n_components=n_features, random_state=RANDOM_NUMBER)
    X_pca = pca.fit_transform(X)

    return X_pca, pca 


def process_missing_vals(X, imputer=None):
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        X_imp = imputer.fit_transform(X)
        return X_imp, imputer
    else:
        X_imp = imputer.transform(X)
        return X_imp, imputer

def build_model():
    # high dimensional linear regression
    return RidgeCV(alphas=[0.1, 1.0, 10.0])

def output_submission(test_ids, predictions):
    # create a submission file
    submission = pd.DataFrame({'id': test_ids, 'y': predictions})
    submission.to_csv('submission.csv', index=False)
    print("submission written to file")

def main():
    # load training features
    X_train = pd.read_csv("eth-aml-2025-project-1/X_train.csv")
    # load training targets
    y_train = pd.read_csv("eth-aml-2025-project-1/y_train.csv")
    # load test features
    X_test  = pd.read_csv("eth-aml-2025-project-1/X_test.csv")

    # delete id columns
    test_ids     = X_test['id'].copy()
    X_train_feat = X_train.drop(columns=['id'])
    y_train_trgt = y_train['y']
    X_test_feat  = X_test.drop(columns=['id'])

    # split data set
    # IDEA !! cross validation for more epochs
    X_train, X_val, y_train, y_val = train_test_split(X_train_feat, y_train_trgt, test_size=0.15, random_state=RANDOM_NUMBER)

    # impute missing vals
    X_train_imp, imputer = process_missing_vals(X_train, imputer=None)
    X_val_imp, _      = process_missing_vals(X_val, imputer=imputer)
    X_test_imp, _     = process_missing_vals(X_test_feat, imputer=imputer)

    # detect outliers
    X_train_outliers, y_train_outliers = detect_outliers(X_train_imp, y_train)
    X_val_outliers, y_val_outliers     = detect_outliers(X_val_imp, y_val)
    

    # select features
    X_train_pca,pca = select_features(X_train_outliers, y_train_outliers)
    X_val_pca = pca.transform(X_val_outliers)
    X_test_pca = pca.transform(X_test_imp)


    # scale for high-dim linear regression -> put all the features on the same scale
    scaler = StandardScaler()
    X_train_scaled     = scaler.fit_transform(X_train_pca)
    X_val_scaled   = scaler.transform(X_val_pca)
    X_test_scaled   = scaler.transform(X_test_pca)

    # model
    model = build_model()
    model.fit(X_train_scaled, y_train_outliers)

    # validation
    y_val_pred = model.predict(X_val_scaled)
    print("R^2: ", r2_score(y_val_outliers, y_val_pred))

    # train on all data
    X_complete_imp, _ = process_missing_vals(X_train_feat, imputer=imputer)
    X_complete_outliers, y_complete_outliers = detect_outliers(X_complete_imp, y_train_trgt)
    X_complete_pca,pca_complete = select_features(X_complete_outliers, y_complete_outliers)
    X_test_complete_pca = pca_complete.transform(X_test_imp)

    X_complete_scaled     = scaler.fit_transform(X_complete_pca)
    X_test_complete_scaled  = scaler.transform(X_test_complete_pca)
    model.fit(X_complete_scaled, y_train_outliers)

    # predict test data and write submission
    y_test_pred = model.predict(X_test_complete_scaled)
    output_submission(test_ids, y_test_pred)


if __name__ == "__main__":
    main()