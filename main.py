import re
import os.path
import argparse
import logging
from six import iteritems
import numpy as np

from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.models import load_model

from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

from utils import load_data, Embeds, Logger
from prepare_data import calc_text_uniq_words, clean_text, convert_text2seq, get_embedding_matrix, clean_seq, split_data, get_bow
from models import get_cnn, get_lstm, get_concat_model, save_predictions, get_tfidf, get_most_informative_features, get_dense_model
from train import train, continue_train, Params
from metrics import calc_metrics, get_metrics, print_metrics


def get_kwargs(kwargs):
    parser = argparse.ArgumentParser(description='-f TRAIN_FILE -t TEST_FILE -o OUTPUT_FILE -e EMBEDS_FILE [-l LOGGER_FILE] [--swear-words SWEAR_FILE] [--wrong-words WRONG_WORDS_FILE] [--warm-start FALSE] [--format-embeds FALSE]')
    parser.add_argument('-f', '--train', dest='train', action='store', help='/path/to/trian_file', type=str)
    parser.add_argument('-t', '--test', dest='test', action='store', help='/path/to/test_file', type=str)
    parser.add_argument('-o', '--output', dest='output', action='store', help='/path/to/output_file', type=str)
    parser.add_argument('-e', '--embeds', dest='embeds', action='store', help='/path/to/embeds_file', type=str)
    parser.add_argument('-l', '--logger', dest='logger', action='store', help='/path/to/log_file', type=str, default=None)
    parser.add_argument('--swear-words', dest='swear_words', action='store', help='/path/to/swear_words_file', type=str, default=None)
    parser.add_argument('--wrong-words', dest='wrong_words', action='store', help='/path/to/wrong_words_file', type=str, default=None)
    parser.add_argument('--warm-start', dest='warm_start', action='store_true')
    parser.add_argument('--format-embeds', dest='format_embeds', action='store', help='file | json | pickle | binary', type=str, default='file')
    parser.add_argument('--config', dest='config', action='store', help='/path/to/config.json', type=str, default=None)
    parser.add_argument('--train-clear', dest='train_clear', action='store', help='/path/to/save_train_clear_file', type=str, default='data/train_clear.csv')
    parser.add_argument('--test-clear', dest='test_clear', action='store', help='/path/to/save_test_clear_file', type=str, default='data/test_clear.csv')
    parser.add_argument('--output-dir', dest='output_dir', action='store', help='/path/to/dir', type=str, default='.')
    for key, value in iteritems(parser.parse_args().__dict__):
        kwargs[key] = value


def main(*kargs, **kwargs):
    get_kwargs(kwargs)
    train_fname = kwargs['train']
    test_fname = kwargs['test']
    result_fname = kwargs['output']
    embeds_fname = kwargs['embeds']
    logger_fname = kwargs['logger']
    swear_words_fname = kwargs['swear_words']
    wrong_words_fname = kwargs['wrong_words']
    warm_start = kwargs['warm_start']
    format_embeds = kwargs['format_embeds']
    config = kwargs['config']
    train_clear = kwargs['train_clear']
    test_clear = kwargs['test_clear']
    output_dir = kwargs['output_dir']

    model_file = {
        'cnn': os.path.join(output_dir, 'cnn.h5'),
        'lstm': os.path.join(output_dir, 'lstm.h5'),
        'concat': os.path.join(output_dir, 'concat.h5'),
        'lr': os.path.join(output_dir, '{}_logreg.bin'),
        'catboost': os.path.join(output_dir, '{}_catboost.bin')
    }

    # ====Create logger====
    logger = Logger(logging.getLogger(), logger_fname)

    # ====Load data====
    logger.info('Loading data...')
    train_df = load_data(train_fname)
    test_df = load_data(test_fname)

    target_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    num_classes = len(target_labels)

    # ====Load additional data====
    logger.info('Loading additional data...')
    swear_words = load_data(swear_words_fname, func=lambda x: set(x.T[0]), header=None)
    wrong_words_dict = load_data(wrong_words_fname, func=lambda x: {val[0] : val[1] for val in x})

    tokinizer = RegexpTokenizer(r'\w+')
    regexps = [re.compile("([a-zA-Z]+)([0-9]+)"), re.compile("([0-9]+)([a-zA-Z]+)")]

    # ====Load word vectors====
    logger.info('Loading embeddings...')
    embed_dim = 300
    embeds = Embeds(embeds_fname, 'fasttext', format=format_embeds)

    # ====Clean texts====
    logger.info('Cleaning text...')
    if warm_start:
        logger.info('Use warm start...')
    else:
        train_df['comment_text_clear'] = clean_text(train_df['comment_text'], tokinizer, wrong_words_dict, swear_words, regexps)
        test_df['comment_text_clear'] = clean_text(test_df['comment_text'], tokinizer, wrong_words_dict, swear_words, regexps)
        train_df.to_csv(train_clear, index=False)
        test_df.to_csv(test_clear, index=False)

    # ====Calculate maximum seq length====
    logger.info('Calc text length...')
    train_df.fillna('unknown', inplace=True)
    test_df.fillna('unknown', inplace=True)
    train_df['text_len'] = train_df['comment_text_clear'].apply(lambda words: len(words.split()))
    test_df['text_len'] = test_df['comment_text_clear'].apply(lambda words: len(words.split()))
    max_seq_len = np.round(train_df['text_len'].mean() + 3*train_df['text_len'].std()).astype(int)
    logger.debug('Max seq length = {}'.format(max_seq_len))

    # ====Prepare data to NN====
    logger.info('Converting texts to sequences...')
    max_words = 100000

    train_df['comment_seq'], test_df['comment_seq'], word_index = convert_text2seq(train_df['comment_text_clear'].tolist(), test_df['comment_text_clear'].tolist(), max_words, max_seq_len, lower=True, char_level=False, uniq=True)
    logger.debug('Dictionary size = {}'.format(len(word_index)))

    logger.info('Preparing embedding matrix...')
    embedding_matrix, words_not_found = get_embedding_matrix(embed_dim, embeds, max_words, word_index)
    logger.debug('Embedding matrix shape = {}'.format(np.shape(embedding_matrix)))
    logger.debug('Number of null word embeddings = {}'.format(np.sum(np.sum(embedding_matrix, axis=1) == 0)))

    logger.info('Deleting unknown words from seq...')
    train_df['comment_seq'] = clean_seq(train_df['comment_seq'], embedding_matrix, max_seq_len)
    test_df['comment_seq'] = clean_seq(test_df['comment_seq'], embedding_matrix, max_seq_len)

    # ====Train/test split data====
    x = np.array(train_df['comment_seq'].tolist())
    y = np.array(train_df[target_labels].values)
    x_train_nn, x_test_nn, y_train_nn, y_test_nn, train_idxs, test_idxs = split_data(x, y, test_size=0.2, shuffle=True, random_state=42)
    test_df_seq = np.array(test_df['comment_seq'].tolist())
    y_nn = []
    logger.debug('X shape = {}'.format(np.shape(x_train_nn)))

    # ====Train models====

    params = Params(config)

    cnn = get_cnn(embedding_matrix,
                    num_classes,
                    embed_dim,
                    max_seq_len,
                    num_filters=params.get('cnn').get('num_filters'),
                    l2_weight_decay=params.get('cnn').get('l2_weight_decay'),
                    dropout_val=params.get('cnn').get('dropout_val'),
                    dense_dim=params.get('cnn').get('dense_dim'),
                    add_sigmoid=True,
                    train_embeds=params.get('cnn').get('train_embeds'))
    lstm = get_lstm(embedding_matrix,
                        num_classes,
                        embed_dim,
                        max_seq_len,
                        l2_weight_decay=params.get('lstm').get('l2_weight_decay'),
                        lstm_dim=params.get('lstm').get('lstm_dim'),
                        dropout_val=params.get('lstm').get('dropout_val'),
                        dense_dim=params.get('lstm').get('dense_dim'),
                        add_sigmoid=True,
                        train_embeds=params.get('lstm').get('train_embeds'))
    concat = get_concat_model(embedding_matrix,
                                  num_classes,
                                  embed_dim,
                                  max_seq_len,
                                  num_filters=params.get('concat').get('num_filters'),
                                  l2_weight_decay=params.get('concat').get('l2_weight_decay'),
                                  lstm_dim=params.get('concat').get('lstm_dim'),
                                  dropout_val=params.get('concat').get('dropout_val'),
                                  dense_dim=params.get('concat').get('dense_dim'),
                                  add_sigmoid=True,
                                  train_embeds=params.get('concat').get('train_embeds'))

    models = []
    for model_label in params.get('models'):
        if model_label == 'cnn':
            models.append([model_label, cnn])
        elif model_label == 'lstm':
            models.append([model_label, lstm])
        elif model_label == 'concat':
            models.append([model_label, concat])
        else:
            raise ValueError('Invalid model {}. Model hasn`t defined.'.format(model_label))

    for i in range(models):
        model_label, model = models[i]
        logger.info("training {} ...".format(model_label))
        if params.get(model_label).get('warm_start') and os.path.exists(params.get(model_label).get('model_file')):
            logger.info('{} warm starting...'.format(model_label))
            model = load_model(params.get(model_label).get('model_file'))
            models[i][1] = model
        else:
            hist = train(x_train_nn,
                         y_train_nn,
                         model,
                         batch_size=params.get(model_label).get('batch_size'),
                         num_epochs=params.get(model_label).get('num_epochs'),
                         learning_rate=params.get(model_label).get('learning_rate'),
                         early_stopping_delta=params.get(model_label).get('early_stopping_delta'),
                         early_stopping_epochs=params.get(model_label).get('early_stopping_epochs'),
                         use_lr_strategy=params.get(model_label).get('use_lr_strategy'),
                         lr_drop_koef=params.get(model_label).get('lr_drop_koef'),
                         epochs_to_drop=params.get(model_label).get('epochs_to_drop'),
                         logger=logger)
        y_nn.append(model.predict(x_test_nn))
        save_predictions(test_df, model.predict(test_df_seq), target_labels, model_label)
        metrics = get_metrics(y_test_nn, y_nn[-1], target_labels)
        logger.debug('{} metrics:\n{}'.format(model_label, print_metrics(metrics)))
        model.save(model_file[model_label])


    # TFIDF + LogReg
    logger.info('training LogReg over tfidf...')
    train_tfidf, val_tfidf, test_tfidf, word_tfidf, char_tfidf = get_tfidf(train_df['comment_text_clear'].values[train_idxs],
                                                                           train_df['comment_text_clear'].values[test_idxs],
                                                                           test_df['comment_text_clear'].values)

    models_lr = []
    metrics_lr = {}
    y_tfidf = []
    for i, label in enumerate(target_labels):
        model = LogisticRegression(C=4.0, solver='sag', max_iter=1000, n_jobs=16)
        model.fit(train_tfidf, y_train_nn[:, i])
        y_tfidf.append(model.predict_proba(val_tfidf)[:,1])
        test_df['tfidf_{}'.format(label)] = model.predict_proba(test_tfidf)[:,1]
        metrics_lr[label] = calc_metrics(y_test_nn[:, i], y_tfidf[-1])
        models_lr.append(model)
        joblib.dump(model, model_file['lr'].format(label))
    metrics_lr['Avg logloss'] = np.mean([metric['Logloss'] for label, metric in metrics_lr.items()])
    logger.debug('LogReg(TFIDF) metrics:\n{}'.format(print_metrics(metrics_lr)))

    # Bow for catboost
    if params.get('catboost').get('add_bow'):
        top_pos_words = []
        top_neg_words = []
        for i in range(num_classes):
            top_pos_words.append([])
            top_neg_words.append([])
            top_pos_words[-1], top_neg_words[-1] = get_most_informative_features([word_tfidf, char_tfidf], models_lr[i], n=params.get('catboost').get('bow_top'))

        top_pos_words = list(set(np.concatenate([[val for score, val in top] for top in top_pos_words])))
        top_neg_words = list(set(np.concatenate([[val for score, val in top] for top in top_neg_words])))
        top = list(set(np.concatenate([top_pos_words, top_neg_words])))
        train_bow = get_bow(train_df['comment_text_clear'].values[train_idxs], top)
        val_bow = get_bow(train_df['comment_text_clear'].values[test_idxs], top)
        test_bow = get_bow(test_df['comment_text_clear'].values, top)
        logger.debug('Count bow words = {}'.format(len(top)))

    # Meta catboost
    logger.info('training catboost as metamodel...')
    train_df['text_unique_len'] = train_df['comment_text_clear'].apply(calc_text_uniq_words)
    test_df['text_unique_len'] = test_df['comment_text_clear'].apply(calc_text_uniq_words)

    train_df['text_unique_koef'] = train_df['text_unique_len'] / train_df['text_len']
    test_df['text_unique_koef'] = test_df['text_unique_len'] / test_df['text_len']

    text_len_features = train_df[['text_len', 'text_unique_len', 'text_unique_koef']].values[test_idxs]

    x_train_catboost = []
    y_train_catboost = y_test_nn
    features = y_nn
    features.extend([text_len_features, np.array(y_tfidf).T])
    if params.get('catboost').get('add_bow'):
        features.append(val_bow)
    for feature in zip(*features):
        x_train_catboost.append(np.concatenate(feature))

    models_cb = []
    metrics_cb = {}
    x_train_cb, x_val_cb, y_train_cb, y_val_cb = train_test_split(x_train_catboost, y_train_catboost, test_size=0.20, random_state=42)
    for i, label in enumerate(target_labels):
        model = CatBoostClassifier(loss_function='Logloss',
                                   iterations=params.get('catboost').get('iterations'),
                                   depth=params.get('catboost').get('depth'),
                                   rsm=params.get('catboost').get('rsm'),
                                   learning_rate=params.get('catboost').get('learning_rate'),
                                   device_config=params.get('catboost').get('device_config'))
        model.fit(x_train_cb, y_train_cb[:, i], eval_set=(x_val_cb, y_val_cb[:, i]), use_best_model=True)
        y_hat_cb = model.predict_proba(x_val_cb)
        metrics_cb[label] = calc_metrics(y_val_cb[:, i], y_hat_cb[:, 1])
        models_cb.append(model)
        joblib.dump(model, model_file['catboost'].format(label))
    metrics_cb['Avg logloss'] = np.mean([metric['Logloss'] for label,metric in metrics_cb.items()])
    logger.debug('CatBoost metrics:\n{}'.format(print_metrics(metrics_cb)))

    # ====Predict====
    logger.info('Applying models...')
    text_len_features = test_df[['text_len', 'text_unique_len', 'text_unique_koef']].values
    y_tfidf_test = test_df[['tfidf_{}'.format(label) for label in target_labels]].values
    x_test_cb = []
    features = []
    for model_label, _ in models:
        features.append(test_df[['{}_{}'.format(model_label, label) for label in target_labels]].values)
    features.extend([text_len_features, y_tfidf_test])
    if params.get('catboost').get('add_bow'):
        features.append(test_bow)
    for feature in tqdm(zip(*features)):
        x_test_cb.append(np.concatenate(feature))

    for label, model in zip(target_labels, models_cb):
        pred = model.predict_proba(x_test_cb)
        test_df[label] = np.array(list(pred))[:, 1]

    # ====Save results====
    logger.info('Saving results...')
    test_df[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_csv(result_fname, index=False, header=True)


if __name__=='__main__':
    main()
