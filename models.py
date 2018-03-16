import numpy as np

from keras.layers import Bidirectional, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Input, Embedding, Dropout, Activation, CuDNNGRU, CuDNNLSTM
from keras.layers import GaussianNoise, SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_is_fitted

from utils.constants import MAX_FEATURES, EMBBEDINGS_SIZE, MAX_LEN
from utils.custom_layers import Attention


def _time_bn_elu():
    def func(x):
        x = TimeDistributed(BatchNormalization())(x)
        x = Activation('elu')(x)
        return x

    return func


def _bn_elu():
    def func(x):
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        return x

    return func


def gru(embedding_matrix, spatial_dropout=0., dropout_dense=0., weight_decay=0.):
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(MAX_FEATURES, EMBBEDINGS_SIZE, weights=[embedding_matrix], trainable=False)(inp)
    x = GaussianNoise(stddev=0.15)(x)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = _time_bn_elu()(x)

    x = SpatialDropout1D(spatial_dropout)(x)

    x = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x = _time_bn_elu()(x)

    x = Attention(MAX_LEN)(x)

    x = Dropout(dropout_dense)(x)
    x = Dense(128, kernel_regularizer=l2(weight_decay))(x)
    x = _bn_elu()(x)
    out = Dense(6, kernel_regularizer=l2(weight_decay),
                activation="sigmoid")(x)

    return inp, out


def lstm(embedding_matrix, spatial_dropout=0., dropout_dense=0., weight_decay=0.):
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(MAX_FEATURES, EMBBEDINGS_SIZE, weights=[embedding_matrix], trainable=False)(inp)
    x = GaussianNoise(stddev=0.15)(x)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = _time_bn_elu()(x)

    x = SpatialDropout1D(spatial_dropout)(x)

    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = _time_bn_elu()(x)

    x = Attention(MAX_LEN)(x)

    x = Dropout(dropout_dense)(x)
    x = Dense(128, kernel_regularizer=l2(weight_decay))(x)
    x = _bn_elu()(x)
    out = Dense(6, kernel_regularizer=l2(weight_decay),
                activation="sigmoid")(x)

    return inp, out


def textcnn(embedding_matrix, dropout_dense=0., weight_decay=0.):
    inp = Input(shape=(MAX_LEN,))
    x = Embedding(MAX_FEATURES, EMBBEDINGS_SIZE,
                  weights=[embedding_matrix],
                  trainable=False)(inp)
    x = GaussianNoise(stddev=0.05)(x)

    x = Conv1D(filters=256,
               kernel_size=5,
               dilation_rate=2,
               padding='same')(x)
    x = _bn_elu()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(filters=512,
               kernel_size=5,
               dilation_rate=2,
               padding='same')(x)
    x = _bn_elu()(x)

    x = Attention(MAX_LEN // 2)(x)

    x = Dense(128, kernel_regularizer=l2(weight_decay))(x)
    x = _bn_elu()(x)
    x = Dropout(dropout_dense)(x)
    out = Dense(6, use_bias=True, activation="sigmoid")(x)
    return inp, out


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 C=3.15,
                 dual=False,
                 solver='newton-cg',
                 max_iter=1000,
                 tol=0.00001,
                 n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.solver = solver
        self.tol = tol

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C,
                                       dual=self.dual,
                                       class_weight='balanced',
                                       solver=self.solver,
                                       max_iter=self.max_iter,
                                       tol=self.tol,
                                       n_jobs=self.n_jobs).fit(x_nb, y)
        return self


def get_model(name='gru', **params):
    MODELS_DICT = {
        'gru': gru,
        'lstm': lstm,
        'textcnn': textcnn,
    }

    model_fn = MODELS_DICT[name]
    inp, out = model_fn(**params)

    model = Model(inputs=inp, outputs=out)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001, amsgrad=True),
                  metrics=['binary_accuracy'])

    return model
