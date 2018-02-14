import re
import random
import numpy as np
from tqdm import tqdm

from nltk.tokenize import RegexpTokenizer

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46371
def substitute_repeats_fixed_len(text, nchars, ntimes=3):
    return re.sub(r"(\S{{{}}})(\1{{{},}})".format(nchars, ntimes-1), r"\1", text)


def substitute_repeats(text, ntimes=3):
    for nchars in range(1, 20):
        text = substitute_repeats_fixed_len(text, nchars, ntimes)
    return text


def split_text_and_digits(text, regexps):
    for regexp in regexps:
        result = regexp.match(text)
        if result is not None:
            return ' '.join(result.groups())
    return text


def clean_text(df, wrong_words_dict, autocorrect=True):
    df.fillna("__NA__", inplace=True)
    tokinizer = RegexpTokenizer(r'\w+')
    regexps = [re.compile("([a-zA-Z]+)([0-9]+)"), re.compile("([0-9]+)([a-zA-Z]+)")]
    texts = df.tolist()
    result = []
    for text in tqdm(texts):
        tokens = tokinizer.tokenize(text.lower())
        tokens = [split_text_and_digits(token, regexps) for token in tokens]
        tokens = [substitute_repeats(token, 3) for token in tokens]
        text = ' '.join(tokens)
        if autocorrect:
            for wrong, right in wrong_words_dict.items():
                text = text.replace(wrong, right)
        result.append(text)
    return result


def uniq_words_in_text(text):
    return ' '.join(list(set(text.split())))


def delete_unknown_words(text, embeds):
    return ' '.join([word for word in text.split() if word in embeds])


def convert_text2seq(train_texts, test_texts, max_words, max_seq_len, embeds, lower=True, char_level=False, uniq=False, use_only_exists_words=False):
    tokenizer = Tokenizer(num_words=max_words, lower=lower, char_level=char_level)
    texts = train_texts + test_texts
    if uniq:
        texts = [uniq_words_in_text(text) for text in texts]
    if use_only_exists_words:
        texts = [delete_unknown_words(text, embeds) for text in texts]
    tokenizer.fit_on_texts(texts)
    word_seq_train = tokenizer.texts_to_sequences(train_texts)
    word_seq_test = tokenizer.texts_to_sequences(test_texts)
    word_index = tokenizer.word_index
    word_seq_train = list(sequence.pad_sequences(word_seq_train, maxlen=max_seq_len))
    word_seq_test = list(sequence.pad_sequences(word_seq_test, maxlen=max_seq_len))
    return word_seq_train, word_seq_test, word_index


def split_data_idx(n, test_size=0.2, shuffle=True, random_state=0):
    train_size = 1 - test_size
    idxs = np.arange(n)
    if shuffle:
        random.seed(random_state)
        random.shuffle(idxs)
    return idxs[:int(train_size*n)], idxs[int(train_size*n):]


def split_data(x, y, test_size=0.2, shuffle=True, random_state=0):
    n = len(x)
    train_idxs, test_idxs = split_data_idx(n, test_size, shuffle, random_state)
    return x[train_idxs], x[test_idxs], y[train_idxs], y[test_idxs], train_idxs, test_idxs


def parse_seq(text, type):
    text = re.sub('[^0-9. ]','', text)
    text = re.sub(' +',' ', text)
    text = text.strip().split()
    return np.array([type(val) for val in text])
