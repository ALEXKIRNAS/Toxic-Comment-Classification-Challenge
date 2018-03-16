import concurrent.futures

import click
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from models import NbSvmClassifier
from utils.constants import RANDOM_SEED, CLASS_NAMES
from utils.data_loader import tf_idf_vectors

cv_params = [
        {'C': 0.7},
        {'C': 0.25},
        {'C': 0.27},
        {'C': 0.25},
        {'C': 0.25},
        {'C': 0.25},
    ]
train_word_features, test_word_features, train, test = None, None, None, None


def training(train_indices, val_indices, class_name, params):
    classifier = NbSvmClassifier(**params)

    csr = train_word_features.tocsr()
    X_train = csr[train_indices]
    y_train = np.array(train[class_name])[train_indices]

    X_test = csr[val_indices]
    y_test = np.array(train[class_name])[val_indices]

    classifier.fit(X_train, y_train)

    train_proba = classifier.predict_proba(X_train)[:, 1]
    val_proba = classifier.predict_proba(X_test)[:, 1]
    sub_proba = classifier.predict_proba(test_word_features)[:, 1]

    train_score = roc_auc_score(y_train, train_proba)
    val_score = roc_auc_score(y_test, val_proba)

    return train_score, val_score, val_proba, sub_proba, val_indices


@click.command()
@click.option('--train_df_path', default='./input/train.csv')
@click.option('--test_df_path', default='./input/test.csv')
@click.option('--stamp', default='gru_default_text_fast_text_emb')
def main(train_df_path, test_df_path, stamp):
    global train_word_features, test_word_features, train, test

    train = pd.read_csv(train_df_path).fillna(' ')
    test = pd.read_csv(test_df_path).fillna(' ')

    train_word_features, test_word_features = tf_idf_vectors(train, test)
    submission = pd.DataFrame.from_dict({'id': test['id']})
    train_submission = pd.DataFrame.from_dict({'id': train['id']})

    scores = []
    for i, class_name in enumerate(CLASS_NAMES):
        print('Class: %s' % class_name)

        sub_probas = np.zeros(shape=(len(test),))
        train_probas = np.zeros(shape=(len(train),))

        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

        train_scores, val_scores = [], []
        with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:

            futures = (executor.submit(training,
                                       train_indices,
                                       val_indices,
                                       class_name,
                                       cv_params[i])
                       for train_indices, val_indices in kf.split(train))

            for future in concurrent.futures.as_completed(futures):
                train_score, val_score, val_proba, sub_proba, val_indices = future.result()
                train_scores.append(train_score)
                val_scores.append(val_score)

                train_probas[val_indices] += val_proba
                sub_probas += sub_proba / 5.

        scores.append(np.mean(val_scores))
        print('\tTrain ROC-AUC: %s' % np.mean(train_scores))
        print('\tVal ROC-AUC: %s' % np.mean(val_scores))

        submission[class_name] = sub_probas
        train_submission[class_name] = train_probas

        submission.to_csv('%s.csv' % stamp, index=False)
        train_submission.to_csv('%s.csv' % stamp, index=False)

    print('Total: %s' % np.mean(scores))


if __name__ == '__main_':
    main()