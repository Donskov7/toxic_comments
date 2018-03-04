import re
import random
import numpy as np
from tqdm import tqdm

from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer

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


def sparse_to_seq(matrix, maxlen):
    seqs = []
    for i in range(np.shape(matrix)[0]):
        seq = []
        for (_, j), cnt in matrix[i].todok().items():
            for _ in range(cnt):
                seq.append(j+1)
        if len(seq) < maxlen:
            for _ in range(maxlen-len(seq)):
                seq.append(0)
        if len(seq) > maxlen:
            random.shuffle(seq)
            seq = seq[:maxlen]
        seqs.append(seq)
    return seqs


def convert_text2seq(train_texts, test_texts,
                     max_words, max_seq_len, max_char_seq_len, embeds,
                     lower=True, oov_token='__NA__',
                     uniq=False, use_only_exists_words=False):
    texts = train_texts + test_texts
    if uniq:
        texts = [uniq_words_in_text(text) for text in texts]
    if use_only_exists_words:
        texts = [delete_unknown_words(text, embeds) for text in texts]

    # WORD TOKENIZER
    word_tokenizer = Tokenizer(num_words=max_words, lower=lower, char_level=False)
    word_tokenizer.fit_on_texts(texts)

    word_seq_train = word_tokenizer.texts_to_sequences(train_texts)
    word_seq_test = word_tokenizer.texts_to_sequences(test_texts)
    word_index = word_tokenizer.word_index

    word_seq_train = list(sequence.pad_sequences(word_seq_train, maxlen=max_seq_len))
    word_seq_test = list(sequence.pad_sequences(word_seq_test, maxlen=max_seq_len))

    # CHAR TOKENIZER
    char_tokenizer = CountVectorizer(analyzer='char', ngram_range=(3,3), stop_words=None, lowercase=True,
                            max_df=0.9, min_df=0, max_features=max_words)
    char_tokenizer.fit(texts)
    char_sparse_train = char_tokenizer.transform(train_texts)
    char_sparse_test = char_tokenizer.transform(test_texts)

    char_seq_train = sparse_to_seq(char_sparse_train, maxlen=max_char_seq_len)
    char_seq_test = sparse_to_seq(char_sparse_test, maxlen=max_char_seq_len)

    char_index = {key: val+1 for key, val in char_tokenizer.vocabulary_.items()}
    char_index[oov_token] = 0
    char_vocab_len = len(char_index)

    return word_seq_train, word_seq_test, word_index, char_seq_train, char_seq_test, char_index


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
