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
    iso_forest = IsolationForest(n_estimators='auto', contamination=0.069, random_state=RANDOM_NUMBER)
    prediction = iso_forest.fit_predict(X)
    inliners = prediction == 1
    return X.loc[inliners], y.loc[inliners]

def select_features(X, y):
    

    return X


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
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_feat, y_train_trgt, test_size=0.15, random_state=RANDOM_NUMBER)

    # impute missing vals
    X_tr_imp, imputer = process_missing_vals(X_tr, imputer=None)
    X_val_imp, _      = process_missing_vals(X_val, imputer=imputer)
    X_test_imp, _     = process_missing_vals(X_test_feat, imputer=imputer)

    # detect outliers
    X_train_feat, y_train_trgt = detect_outliers(X_train_feat, y_train_trgt)

    # select features
    X_train_feat = select_features(X_train_feat, y_train_trgt)


    # scale for high-dim linear regression
    scaler = StandardScaler()
    X_tr_sc     = scaler.fit_transform(X_tr_imp)
    X_val_sc    = scaler.transform(X_val_imp)
    X_test_sc   = scaler.transform(X_test_imp)

    # model
    model = build_model()
    model.fit(X_tr_sc, y_tr)

    # validation
    y_val_pred = model.predict(X_val_sc)
    print("R^2: ", r2_score(y_val, y_val_pred))

    # train on all data
    X_comp_imp, _ = process_missing_vals(X_train_feat, imputer=imputer)
    X_comp_sc     = scaler.fit_transform(X_comp_imp)
    model.fit(X_comp_sc, y_train_trgt)

    # predict test data and write submission
    y_test_pred = model.predict(X_test_sc)
    output_submission(test_ids, y_test_pred)


if __name__ == "__main__":
    main()