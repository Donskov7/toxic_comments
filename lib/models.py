from __future__ import absolute_import
import numpy as np
from scipy import sparse

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from keras import regularizers
from keras.regularizers import l1, l2, l1_l2
from keras.utils import multi_gpu_model
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Concatenate, Embedding, BatchNormalization
from keras.layers import Dense, Bidirectional, LSTM, GRU, CuDNNLSTM, CuDNNGRU
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D

try:
    from utils import GlobalZeroMaskedAveragePooling1D, GlobalSumPooling1D
except ImportError:
    from .utils import GlobalZeroMaskedAveragePooling1D, GlobalSumPooling1D


def cnn(embedding_matrix, char_matrix, num_classes, max_seq_len, max_ll3_seq_len,
        num_filters=64, l2_weight_decay=0.0001, dropout_val=0.5,
        dense_dim=32, add_sigmoid=True, train_embeds=False, gpus=0,
        n_cnn_layers=1, pool='max', add_embeds=False):
    if pool == 'max':
        Pooling = MaxPooling1D
        GlobalPooling = GlobalMaxPooling1D
    elif pool == 'avg':
        Pooling = AveragePooling1D
        GlobalPooling = GlobalAveragePooling1D
    input_ = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       trainable=train_embeds)(input_)
    x = embeds
    for i in range(n_cnn_layers-1):
        x = Conv1D(num_filters, 7, activation='relu', padding='same')(x)
        x = Pooling(2)(x)
    x = Conv1D(num_filters, 7, activation='relu', padding='same')(x)
    x = GlobalPooling()(x)
    if add_embeds:
        x1 = Conv1D(num_filters, 7, activation='relu', padding='same')(embeds)
        x1 = GlobalPooling()(x1)
        x = Concatenate()([x, x1])
    x = BatchNormalization()(x)
    x = Dropout(dropout_val)(x)
    x = Dense(dense_dim, activation='relu', kernel_regularizer=regularizers.l2(l2_weight_decay))(x)
    if add_sigmoid:
        x = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=input_, outputs=x)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model


def _get_regularizer(regularizer_name, weight):
    if regularizer_name is None:
        return None
    if regularizer_name == 'l1':
        return l1(weight)
    if regularizer_name == 'l2':
        return l2(weight)
    if regularizer_name == 'l1_l2':
        return l1_l2(weight)
    return None

def rnn(embedding_matrix, char_matrix, num_classes,  max_seq_len, max_ll3_seq_len,
        l2_weight_decay=0.0001, rnn_dim=100, dropout_val=0.3,
        dense_dim=32, n_rnn_layers=1, n_dense_layers=1, add_sigmoid=True,
        train_embeds=False, gpus=0, rnn_type='lstm', mask_zero=True,
        kernel_regularizer=None, recurrent_regularizer=None,
        activity_regularizer=None, dropout=0.0, recurrent_dropout=0.0,
        pool='max', add_embeds=True, return_state=False):
    GlobalPool = {
        'avg': GlobalZeroMaskedAveragePooling1D,
        'max': GlobalMaxPooling1D,
        'sum': GlobalSumPooling1D
    }
    rnn_regularizers = {'kernel_regularizer': _get_regularizer(kernel_regularizer, l2_weight_decay),
                        'recurrent_regularizer': _get_regularizer(recurrent_regularizer, l2_weight_decay),
                        'activity_regularizer': _get_regularizer(activity_regularizer, l2_weight_decay)}
    if gpus == 0:
        rnn_regularizers['dropout'] = dropout
        rnn_regularizers['recurrent_dropout'] = recurrent_dropout
    if rnn_type == 'lstm':
        RNN = LSTM # CuDNNLSTM if gpus > 0 else LSTM
    elif rnn_type == 'gru':
        RNN = GRU # CuDNNGRU if gpus > 0 else GRU
    mask_zero = mask_zero and gpus == 0

    input_ = Input(shape=(max_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       mask_zero=mask_zero,
                       trainable=train_embeds)(input_)
    x = embeds
    for _ in range(n_rnn_layers-1):
        x = Bidirectional(RNN(rnn_dim, return_sequences=True, **rnn_regularizers))(x)
    x = Bidirectional(RNN(rnn_dim, return_sequences=False, return_state=return_state, **rnn_regularizers))(x)
    if return_state:
        x = Concatenate()(x)
    if add_embeds:
        embeds2 = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       mask_zero=False,
                       trainable=train_embeds)(input_)
        if isinstance(pool, list) and len(pool) > 1:
            to_concat = []
            for p in pool:
                to_concat.append(GlobalPool[p]()(embeds2))
            x1 = Concatenate()(to_concat)
        else:
            x1 = GlobalPool[pool]()(embeds2)
        x = Concatenate()([x, x1])
    x = BatchNormalization()(x)
    x = Dropout(dropout_val)(x)
    for _ in range(n_dense_layers-1):
        x = Dense(dense_dim, activation="relu")(x)
        x = Dropout(dropout_val)(x)
    x = Dense(dense_dim, activation="relu", kernel_regularizer=regularizers.l2(l2_weight_decay))(x)
    if add_sigmoid:
        x = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=input_, outputs=x)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model


def dense(embedding_matrix, ll3_matrix, num_classes, max_seq_len, max_ll3_seq_len,
          dense_dim=100, n_layers=10, concat=0, dropout_val=0.5,
          l2_weight_decay=0.0001, pool='max', add_sigmoid=True,
          train_embeds=False, gpus=0, add_ll3=True):
    GlobalPool = {
        'avg': GlobalZeroMaskedAveragePooling1D,
        'max': GlobalMaxPooling1D,
        'sum': GlobalSumPooling1D
    }

    input_ = Input(shape=(max_seq_len,))
    input2_ = Input(shape=(max_ll3_seq_len,))
    embeds = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=max_seq_len,
                       trainable=train_embeds)(input_)
    ll3_embeds = Embedding(ll3_matrix.shape[0],
                       ll3_matrix.shape[1],
                       weights=[ll3_matrix],
                       input_length=max_ll3_seq_len,
                       trainable=True)(input2_)
    if isinstance(pool, list) and len(pool) > 1:
        to_concat = []
        for p in pool:
            to_concat.append(GlobalPool[p]()(embeds))
            if add_ll3:
                to_concat.append(GlobalPool[p]()(ll3_embeds))
        x = Concatenate()(to_concat)
    else:
        x = GlobalPool[pool]()(embeds)
        if add_ll3:
            x1 = GlobalPool[pool]()(ll3_embeds)
            x = Concatenate()([x, x1])
    x = BatchNormalization()(x)
    prev = []
    for i in range(n_layers):
        if concat > 0:
            if i == 0:
                prev.append(x)
                continue
            elif i % concat == 0:
                prev.append(x)
                x = Concatenate(axis=-1)(prev)
        x = Dense(dense_dim, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_val)(x)
    output_ = Dense(dense_dim, activation="relu", kernel_regularizer=regularizers.l2(l2_weight_decay))(x)
    if add_sigmoid:
        output_ = Dense(num_classes, activation="sigmoid")(output_)
    if add_ll3:
        model = Model(inputs=[input_, input2_], outputs=output_)
    else:
        model = Model(inputs=input_, outputs=output_)
    if gpus > 0:
        model = multi_gpu_model(model, gpus=gpus)
    return model


class TFIDF(object):
    def __init__(self, target_labels, *args, **kwargs):
        self.target_labels = target_labels
        self.n_classes = len(target_labels)
        params = {
            'C': 4.0,
            'solver': 'sag',
            'max_iter': 1000,
            'n_jobs': 16
        }
        params.update(kwargs)
        self.models = [LogisticRegression(*args, **params) for _ in range(self.n_classes)]
        self.word_tfidf = None
        self.char_tfidf = None

    def fit(self, X, y, max_features=50000):
        assert np.shape(y)[1] == self.n_classes
        x_tfidf = self.fit_tfidf(X, max_features)
        for i, model in enumerate(self.models):
            model.fit(x_tfidf, y[:, i])

    def predict(self, X):
        y = []
        x_tfidf = self.transform_tfidf(X)
        for i, model in enumerate(self.models):
            y.append(model.predict(x_tfidf))
        return np.transpose(y)

    def fit_tfidf(self, X, max_features):
        self.word_tfidf = TfidfVectorizer(max_features=max_features, analyzer='word', lowercase=True, ngram_range=(1, 3), token_pattern='[a-zA-Z0-9]')
        self.char_tfidf = TfidfVectorizer(max_features=max_features, analyzer='char', lowercase=True, ngram_range=(1, 5), token_pattern='[a-zA-Z0-9]')

        tfidf_word = self.word_tfidf.fit_transform(X)
        tfidf_char = self.char_tfidf.fit_transform(X)

        return sparse.hstack([tfidf_word, tfidf_char])

    def transform_tfidf(self, X):
        assert self.word_tfidf != None and self.char_tfidf != None
        tfidf_word = self.word_tfidf.transform(X)
        tfidf_char = self.char_tfidf.transform(X)

        return sparse.hstack([tfidf_word, tfidf_char])


class CatBoost(object):
    def __init__(self, target_labels, *args, **kwargs):
        self.target_labels = target_labels
        self.n_classes = len(target_labels)
        self.models = [CatBoostClassifier(*args, **kwargs) for _ in range(self.n_classes)]

    def fit(self, X, y, eval_set=None, use_best_model=True):
        assert np.shape(y)[1] == self.n_classes
        for i, model in enumerate(self.models):
            if eval_set is not None:
                eval_set_i = (eval_set[0], eval_set[1][:, i])
            else:
                eval_set_i = None
            model.fit(X, y[:, i], eval_set=eval_set_i, use_best_model=use_best_model)

    def predict(self, X):
        y = []
        for i, model in enumerate(self.models):
            y.append(model.predict(X))
        return np.transpose(y)

    def predict_proba(self, X):
        y = []
        for i, model in enumerate(self.models):
            y.append(model.predict_proba(X)[:, 1])
        return np.transpose(y)


def save_predictions(df, predictions, target_labels, additional_name=None):
    for i, label in enumerate(target_labels):
        if additional_name is not None:
            label = '{}_{}'.format(additional_name, label)
        df[label] = predictions[:, i]
