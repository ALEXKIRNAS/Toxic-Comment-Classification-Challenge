import os

import click
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models import get_model
from utils.callbacks import get_model_callbacks
from utils.constants import BATCH_SIZE, EPOCHES, CLASS_NAMES
from utils.data_loader import load_data
from utils.validation import prepare_data_cv


@click.command()
@click.option('--train_df_path', default='./input/train.csv')
@click.option('--test_df_path', default='./input/test.csv')
@click.option('--embedings_file', default='./input/crawl-300d-2M.vec')
@click.option('--model_name', default='gru')
@click.option('--stamp', default='gru_default_text_fast_text_emb')
def main(train_df_path, test_df_path, embedings_file, model_name, stamp):
  experiment_path = './experiments/%s' % stamp

  x_train, targets_train, x_test, train_submission, submission, embedings = load_data(train_df_path, test_df_path,
                                                                                      embedings_file)
  (kfold_data, X_test) = prepare_data_cv(x_train, targets_train, x_test)

  train_probas = np.zeros(shape=(x_train.shape[0], 6))
  test_probas = np.zeros(shape=(x_test.shape[0], 6))

  models_roc = []
  models_train_roc = []

  for idx, data in enumerate(tqdm(kfold_data)):
    X_train, y_train, X_valid, y_valid, val_indices = data

    model = get_model(model_name,
                      embedding_matrix=embedings,
                      dropout_dense=0.4,
                      weight_decay=1e-4)
    callbacks = get_model_callbacks(save_dir=os.path.join(experiment_path, 'fold_%02d' % idx))

    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHES,
              validation_data=(X_valid, y_valid),
              shuffle=True,
              callbacks=callbacks, verbose=1)

    model.load_weights(filepath=os.path.join(experiment_path, ('fold_%02d/model/model_weights.hdf5' % idx)))

    proba = model.predict(X_train, batch_size=BATCH_SIZE * 2)
    proba_val = model.predict(X_valid, batch_size=BATCH_SIZE * 2)
    proba_test = model.predict(x_test, batch_size=BATCH_SIZE * 2)

    models_roc.append(roc_auc_score(y_valid, proba_val))
    models_train_roc.append(roc_auc_score(y_train, proba))

    train_probas[val_indices] += proba_val
    test_probas += proba_test / 5.

    print('Train ROC AUC:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_train_roc),
                                                                       np.std(models_train_roc),
                                                                       np.min(models_train_roc),
                                                                       np.max(models_train_roc)))

    print('Val ROC AUC:\nMean: %f\nStd: %f\nMin: %f\nMax: %f\n\n' % (np.mean(models_roc),
                                                                     np.std(models_roc),
                                                                     np.min(models_roc),
                                                                     np.max(models_roc)))

  for i, cls_name in enumerate(CLASS_NAMES):
    train_submission[cls_name] = train_probas[:, i]
  train_submission.to_csv('./csv/train_%s.csv' % stamp, index=False)

  for i, cls_name in enumerate(CLASS_NAMES):
    submission[cls_name] = test_probas[:, i]
  submission.to_csv('./csv/submission_%s.csv' % stamp, index=False)


if __name__ == '__main__':
  main()
