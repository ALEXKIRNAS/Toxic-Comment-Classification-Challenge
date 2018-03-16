import os

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, batch_size=512, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print(" - ROC-AUC score: %.6f \n" % score)


def get_model_callbacks(save_dir):
    stopping = EarlyStopping(monitor='val_loss',
                             min_delta=1e-4,
                             patience=3,
                             verbose=False,
                             mode='min')

    board_path = os.path.join(save_dir, 'board')
    if not os.path.exists(board_path):
        os.makedirs(board_path)

    lr_sheduler = ReduceLROnPlateau(monitor='val_loss',
                                    factor=0.1,
                                    patience=2,
                                    verbose=True,
                                    mode='min',
                                    epsilon=5e-4,
                                    min_lr=1e-5)

    model_path = os.path.join(save_dir, 'model/model_weights.hdf5')
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))

    model_checkpoint = ModelCheckpoint(model_path,
                                       monitor='val_loss',
                                       verbose=False,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='min',
                                       period=1)

    callbacks = [stopping, lr_sheduler, model_checkpoint]
    return callbacks
