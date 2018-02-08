import re
import sys
import json
import logging

import pandas as pd
from tqdm import tqdm
import numpy as np
import gensim

from keras import backend as K
from keras.engine import Layer

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import nirvana_dl
except ImportError:
    pass


def load_data(fname, **kwargs):
    func = kwargs.get('func', None)
    if func is not None:
        del kwargs['func']
    df = pd.read_csv(fname, **kwargs)
    if func is not None:
        return func(df.values)
    return df


class Params(object):
    def __init__(self, config=None):
        self._params = self._common_init()
        config_params = self._load_from_file(config)
        self._update_params(config_params)

    def _load_from_file(self, fname):
        if fname is None:
            return {}
        elif fname == 'nirvana':
            return nirvana_dl.params()
        elif isinstance(fname, dict):
            return fname
        with open(fname) as f:
            return json.loads(f.read())

    def _common_init(self):
        common_params = {
                    'warm_start': False,
                    'model_file': None,
                    'batch_size': 256,
                    'num_epochs': 10,
                    'learning_rate': 0.0001,
                    'use_lr_strategy': True,
                    'lr_drop_koef': 0.9,
                    'epochs_to_drop': 1,
                    'early_stopping_delta': 0.001,
                    'early_stopping_epochs': 4,
                    'l2_weight_decay':0.0001,
                    'dropout_val': 0.5,
                    'dense_dim': 32,
                    'mask_zero': True,
                    'train_embeds': False}

		params = {'models': [],
                  'dense': common_params,
                  'cnn': common_params,
                  'lstm': common_params,
                  'gru': common_params}

        params['dense']['dense_dim'] = 100
        params['dense']['n_layers'] = 10
        params['dense']['concat'] = 0
        params['dense']['pool'] = 'max'

        params['cnn']['num_filters'] = 64
        params['cnn']['pool'] = 'max'
        params['cnn']['n_cnn_layers'] = 1
        params['cnn']['add_embeds'] = False

        params['lstm']['rnn_dim'] = 100
        params['lstm']['n_branches'] = 0
        params['lstm']['n_rnn_layers'] = 1
        params['lstm']['n_dense_layers'] = 1
        params['lstm']['kernel_regularizer'] = None
        params['lstm']['recurrent_regularizer'] = None
        params['lstm']['activity_regularizer'] = None
        params['lstm']['dropout'] = 0.0
        params['lstm']['recurrent_dropout'] = 0.0

        params['gru']['rnn_dim'] = 100
        params['gru']['n_branches'] = 0
        params['gru']['n_rnn_layers'] = 1
        params['gru']['n_dense_layers'] = 1
        params['gru']['kernel_regularizer'] = None
        params['gru']['recurrent_regularizer'] = None
        params['gru']['activity_regularizer'] = None
        params['gru']['dropout'] = 0.0
        params['gru']['recurrent_dropout'] = 0.0

		params['catboost'] = {
                    'add_bow': False,
                    'bow_top': 100,
                    'iterations': 1000,
                    'depth': 6,
                    'rsm': 1,
                    'learning_rate': 0.01,
                    'device_config': None}
        return params

    def _update_params(self, params):
        if params is not None and params:
            for key in params.keys():
                if isinstance(params[key], dict):
                    self._params.setdefault(key, {})
                    self._params[key].update(params[key])
                else:
                    self._params[key] = params[key]

    def get(self, key):
        return self._params.get(key, None)


class Embeds(object):
    def __init__(self, fname, w2v_type='fasttext', format='file'):
        if format in ('json', 'pickle'):
            self.load(fname, format)
        elif w2v_type == 'fasttext':
            self.model = self._read_fasttext(fname)
        elif w2v_type == 'word2vec':
            self.model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=format=='binary')
        else:
            self.model = {}

    def __getitem__(self, key):
        try:
            return self.model[key]
        except KeyError:
            return None

    def __contains__(self, key):
        return self[key] is not None

    def _process_line(self, line):
        line = line.rstrip().split(' ')
        word = line[0]
        vec = line[1:]
        return word, [float(val) for val in vec]

    def _read_fasttext(self, fname):
        with open(fname) as f:
            tech_line = f.readline()
            dict_size, vec_size = self._process_line(tech_line)
            print('dict_size = {}'.format(dict_size))
            print('vec_size = {}'.format(vec_size))
            model = {}
            for line in tqdm(f, file=sys.stdout):
                word, vec = self._process_line(line)
                model[word] = vec
        return model

    def save(self, fname, format='json'):
        if format == 'json':
            with open(fname, 'w') as f:
                json.dump(self.model, f)
        elif format == 'pickle':
            with open(fname, 'wb') as f:
                pickle.dump(self.model, f)
        return self

    def load(self, fname, format='json'):
        if format == 'json':
            with open(fname) as f:
                self.model = json.load(f)
        elif format == 'pickle':
            with open(fname, 'rb') as f:
                self.model = pickle.load(f)
        return self


class Logger(object):
    def __init__(self, logger, fname=None, format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"):
        self.logFormatter = logging.Formatter(format)
        self.rootLogger = logger
        self.rootLogger.setLevel(logging.DEBUG)

        self.consoleHandler = logging.StreamHandler(sys.stdout)
        self.consoleHandler.setFormatter(self.logFormatter)
        self.rootLogger.addHandler(self.consoleHandler)

        if fname is not None:
            self.fileHandler = logging.FileHandler(fname)
            self.fileHandler.setFormatter(self.logFormatter)
            self.rootLogger.addHandler(self.fileHandler)

    def warn(self, message):
        self.rootLogger.warn(message)

    def info(self, message):
        self.rootLogger.info(message)

    def debug(self, message):
        self.rootLogger.debug(message)


class GlobalZeroMaskedAveragePooling1D(Layer):
    def __init__(self, **kwargs):
        super(GlobalZeroMaskedAveragePooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def call(self, x, mask=None):
        mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)
        n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)
        x_mean = K.sum(x, axis=1, keepdims=False) / (n + 1)
        return K.cast(x_mean, 'float32')

    def compute_mask(self, x, mask=None):
        return None


class GlobalSumPooling1D(Layer):
    def __init__(self, **kwargs):
        super(GlobalSumPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def call(self, x, mask=None):
        return K.sum(x, axis=1, keepdims=False)

    def compute_mask(self, x, mask=None):
        return None


def embed_aggregate(seq, embeds, func=np.sum, normalize=False):
    embed_dim = len(embeds[0])
    embed = np.zeros(embed_dim)
    nozeros = 0
    for value in seq:
        if value == 0:
            continue
        embed = func([embed, embeds[value]], axis=0)
        nozeros += 1
    if normalize:
        embed /= nozeros + 1
    return embed


def similarity(seq1, seq2, embeds, pool='max', func=lambda x1, x2: x1 + x2):
    pooling = {
        'max': {'func': np.max},
        'avg': {'func': np.sum, 'normalize': True},
        'sum': {'func': np.sum, 'normalize': False}
    }
    embed1 = embed_aggregate(seq1, embeds, **pooling[pool])
    embed2 = embed_aggregate(seq2, embeds, **pooling[pool])
    return func(embed1, embed2)
