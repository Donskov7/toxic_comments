import os
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
        self._params = self._load_from_file(config)

    def __getitem__(self, key):
        return self._params.get(key, None)

    def _load_from_file(self, fname):
        if fname is None:
            return {}
        elif fname == 'nirvana':
            return nirvana_dl.params()
        elif isinstance(fname, dict):
            return fname
        with open(fname) as f:
            return json.loads(f.read())

    def get(self, key):
        return self._params.get(key, None)


class Embeds(object):
    def __init__(self):
        self.word_index = {}
        self.word_index_reverse = {}
        self.matrix = None
        self.shape = (0, 0)

    def __getitem__(self, key):
        idx = self.word_index.get(key, None)
        if idx is not None:
            return self.matrix[idx]
        return None

    def __contains__(self, key):
        return self[key] is not None

    def _process_line(self, line, embed_dim):
        line = line.rstrip().split(' ')
        word = ' '.join(line[:-embed_dim])
        vec = line[-embed_dim:]
        return word, [float(val) for val in vec]

    def _read_raw_file(self, fname):
        with open(fname) as f:
            tech_line = f.readline()
            word_count, embed_dim = tech_line.rstrip().split()
            word_count = int(word_count) + 1
            embed_dim = int(embed_dim[0])
            print('dict_size = {}'.format(word_count))
            print('embed_dim = {}'.format(embed_dim))
            self.matrix = np.zeros((word_count, embed_dim))
            self.word_index = {}
            self.word_index_reverse = {}
            for i, line in tqdm(enumerate(f), file=sys.stdout):
                word, vec = self._process_line(line, embed_dim)
                self.matrix[i+1] = vec
                self.word_index[word] = i+1
                self.word_index_reverse[i+1] = word
            self.shape = (word_count, embed_dim)
        return self

    def _read_struct_file(self, fname, format):
        if format == 'json':
            data = json.load(open(fname))
        elif format == 'pickle':
            data = pickle.load(open(fname, 'rb'))
        elif format in ('word2vec', 'binary'):
            data = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=format=='binary')
        word_count = len(data) + 1
        embed_dim = len(data.values()[0])
        self.word_index = {}
        self.word_index_reverse = {}
        self.matrix = np.zeros((word_count, embed_dim))
        for i, (word, vec) in enumerate(data.items()):
            self.matrix[i+1] = vec
            self.word_index[word] = i+1
            self.word_index_reverse[i+1] = word
        return self

    def save(self, fname):
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'a') as f:
            f.write('{} {}\n'.format(*self.shape))
            for i,line in enumerate(self.matrix[1:]):
                line = [str(val) for val in line]
                line = ' '.join(line)
                f.write('{} {}\n'.format(self.word_index_reverse[i+1], line))
        return self

    def load(self, fname, format='raw'):
        if format == 'raw':
            self._read_raw_file(fname)
        else:
            self._read_struct_file(fname, format)
        return self

    def set_matrix(self, max_words, word_index):
        words_not_found = []
        word_count = min(max_words, len(word_index)) + 1
        matrix = np.zeros((word_count, self.shape[1]))
        word_index_reverse = {idx : word for word, idx in word_index.items()}
        for word, i in word_index.items():
            if i >= word_count:
                break
            vec = self[word]
            if vec is not None and len(vec) > 0:
                matrix[i] = vec
            else:
                words_not_found.append(word)
        self.matrix = matrix
        self.word_index = word_index
        self.word_index_reverse = word_index_reverse
        self.shape = (word_count, self.shape[1])
        return words_not_found

    def get_matrix(self):
        return self.matrix


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
    embed = np.zeros(embeds.shape[1])
    nozeros = 0
    for value in seq:
        if value == 0 or value > embeds.shape[0]:
            continue
        embed = func([embed, embeds.matrix[value]], axis=0)
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
