from keras.layers import Bidirectional, TimeDistributed
from keras.layers import Dense, Input, Embedding, Dropout, Activation, CuDNNGRU, CuDNNLSTM
from keras.layers import GaussianNoise, SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv1D, MaxPooling1D

from utils.custom_layers import Attention
from utils.constants import MAX_FEATURES, EMBBEDINGS_SIZE, MAX_LEN


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


def gru(embedding_matrix, spatial_dropout, dropout_dense, weight_decay):
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


def lstm(embedding_matrix, spatial_dropout, dropout_dense, weight_decay):
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


def textcnn(embedding_matrix, dropout_dense, weight_decay):
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
