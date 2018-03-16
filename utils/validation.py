import numpy as np
from sklearn.model_selection import KFold


def prepare_data_cv(x_train, targets_train, x_test):
    kfold_data = []
    kf = KFold(n_splits=5, shuffle=True, random_state=0xCAFFE)

    targets_train = np.array(targets_train)
    x_train = np.array(x_train)

    for train_indices, val_indices in kf.split(targets_train):
        X_train_cv = x_train[train_indices]
        y_train_cv = targets_train[train_indices]

        X_val = x_train[val_indices]
        y_val = targets_train[val_indices]
        kfold_data.append((X_train_cv, y_train_cv, X_val, y_val, val_indices))

    X_test = x_test

    return (kfold_data, X_test)
