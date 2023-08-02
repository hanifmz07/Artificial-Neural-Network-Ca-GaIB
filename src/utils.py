import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess_bcancer():
    df_canc = pd.read_csv("bcancer.csv")
    df_canc.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
    encoding = {"B": 0, "M": 1}
    df_canc["diagnosis"] = df_canc["diagnosis"].map(encoding)
    X_canc = df_canc.drop(["diagnosis"], axis=1)
    scaler = MinMaxScaler()
    scaler.fit(X_canc)
    X_transform = scaler.transform(X_canc)
    y_canc = df_canc["diagnosis"].copy()
    X_transform_df = pd.DataFrame(X_transform, columns=X_canc.columns)
    return X_transform, y_canc


def adjust_dimension(X, y):
    X_adj = np.array(X).T
    y_adj = np.array(y).reshape((y.shape[0], 1)).T
    return X_adj, y_adj


def f1_score(y_true, y_pred, show_y=False):
    fp = tp = fn = 0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    categories = np.unique(y_true)
    if show_y:
        print(f"y_true = {y_true}")
        print(f"y_pred = {y_pred}")
    for i in range(len(y_true)):
        if y_true[i] == categories[0] and y_pred[i] == categories[0]:
            tp += 1
        elif y_true[i] == categories[1] and y_pred[i] == categories[0]:
            fp += 1
        elif y_true[i] == categories[0] and y_pred[i] == categories[1]:
            fn += 1
    return (2 * tp) / (2 * tp + fp + fn)
