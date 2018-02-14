from __future__ import absolute_import
import os.path
import argparse
import logging
from six import iteritems
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from keras.models import load_model

from tensorflow.python.client import device_lib

from utils import load_data, Embeds, Logger, Params, embed_aggregate, similarity
from features import catboost_features
from preprocessing import clean_text, convert_text2seq, split_data, parse_seq
from models import cnn, dense, rnn, TFIDF, CatBoost, save_predictions
from train import train
from metrics import get_metrics, print_metrics


def get_kwargs(kwargs):
    parser = argparse.ArgumentParser(description='-f TRAIN_FILE -t TEST_FILE -o OUTPUT_FILE -e EMBEDS_FILE [-l LOGGER_FILE] [--swear-words SWEAR_FILE] [--wrong-words WRONG_WORDS_FILE] [--format-embeds FALSE]')
    parser.add_argument('-f', '--train', dest='train', action='store', help='/path/to/trian_file', type=str)
    parser.add_argument('-t', '--test', dest='test', action='store', help='/path/to/test_file', type=str)
    parser.add_argument('-o', '--output', dest='output', action='store', help='/path/to/output_file', type=str)
    parser.add_argument('-e', '--embeds', dest='embeds', action='store', help='/path/to/embeds_file', type=str)
    parser.add_argument('-c','--config', dest='config', action='store', help='/path/to/config.json', type=str)
    parser.add_argument('-l', '--logger', dest='logger', action='store', help='/path/to/log_file', type=str, default=None)
    parser.add_argument('--mode', dest='mode', action='store', help='preprocess / train / validate / all', type=str, default='all')
    parser.add_argument('--max-words', dest='max_words', action='store', type=int, default=300000)
    parser.add_argument('--use-only-exists-words', dest='use_only_exists_words', action='store_true')
    parser.add_argument('--swear-words', dest='swear_words', action='store', help='/path/to/swear_words_file', type=str, default=None)
    parser.add_argument('--wrong-words', dest='wrong_words', action='store', help='/path/to/wrong_words_file', type=str, default=None)
    parser.add_argument('--format-embeds', dest='format_embeds', action='store', help='file | json | pickle | binary', type=str, default='raw')
    parser.add_argument('--output-dir', dest='output_dir', action='store', help='/path/to/dir', type=str, default='.')
    parser.add_argument('--norm-prob', dest='norm_prob', action='store_true')
    parser.add_argument('--norm-prob-koef', dest='norm_prob_koef', action='store', type=float, default=1)
    parser.add_argument('--gpus', dest='gpus', action='store', help='count GPUs', type=int, default=0)
    for key, value in iteritems(parser.parse_args().__dict__):
        kwargs[key] = value


def main(*kargs, **kwargs):
    get_kwargs(kwargs)
    train_fname = kwargs['train']
    test_fname = kwargs['test']
    result_fname = kwargs['output']
    embeds_fname = kwargs['embeds']
    logger_fname = kwargs['logger']
    mode = kwargs['mode']
    max_words = kwargs['max_words']
    use_only_exists_words = kwargs['use_only_exists_words']
    swear_words_fname = kwargs['swear_words']
    wrong_words_fname = kwargs['wrong_words']
    embeds_format = kwargs['format_embeds']
    config = kwargs['config']
    output_dir = kwargs['output_dir']
    norm_prob = kwargs['norm_prob']
    norm_prob_koef = kwargs['norm_prob_koef']
    gpus = kwargs['gpus']

    seq_col_name = 'comment_seq_use_exist{}_{}k'.format(int(use_only_exists_words), int(max_words/1000))

    model_file = {
        'dense': os.path.join(output_dir, 'dense.h5'),
        'cnn': os.path.join(output_dir, 'cnn.h5'),
        'lstm': os.path.join(output_dir, 'lstm.h5'),
        'lr': os.path.join(output_dir, '{}_logreg.bin'),
        'catboost': os.path.join(output_dir, '{}_catboost.bin')
    }

    # ====Create logger====
    logger = Logger(logging.getLogger(), logger_fname)

    # ====Detect GPUs====
    logger.debug(device_lib.list_local_devices())

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

    # ====Load word vectors====
    logger.info('Loading embeddings...')
    embeds = Embeds().load(embeds_fname, embeds_format)

    # ====Clean texts====
    if mode in ('preprocess', 'all'):
        logger.info('Cleaning text...')
        train_df['comment_text_clear'] = clean_text(train_df['comment_text'], wrong_words_dict, autocorrect=True)
        test_df['comment_text_clear'] = clean_text(test_df['comment_text'], wrong_words_dict, autocorrect=True)
        train_df.to_csv(os.path.join(output_dir, 'train_clear.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_clear.csv'), index=False)

    # ====END OF `PREPROCESS`====
    if mode == 'preprocess':
        return True

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

    if mode in ('all', 'train'):
        train_df[seq_col_name], test_df[seq_col_name], word_index = convert_text2seq(train_df['comment_text_clear'].tolist(),
                                                                                 test_df['comment_text_clear'].tolist(),
                                                                                 max_words,
                                                                                 max_seq_len,
                                                                                 embeds,
                                                                                 lower=True,
                                                                                 char_level=False,
                                                                                 uniq=False,
                                                                                 use_only_exists_words=use_only_exists_words)
        logger.debug('Dictionary size use_exist{} = {}'.format(int(use_only_exists_words), len(word_index)))

        logger.info('Preparing embedding matrix...')
        words_not_found = embeds.set_matrix(max_words, word_index)
        embeds.save(os.path.join(output_dir, 'wiki.embeds.{}k'.format(int(max_words/1000))))

        # ====Get text vector====
        pooling = {
                'max': {'func': np.max},
                'avg': {'func': np.sum, 'normalize': True},
                'sum': {'func': np.sum, 'normalize': False}
        }
        for p in ['max', 'avg', 'sum']:
            train_df['comment_vec_{}'.format(p)] = train_df[seq_col_name].apply(lambda x: embed_aggregate(x, embeds, **pooling[p]))
            test_df['comment_vec_{}'.format(p)] = test_df[seq_col_name].apply(lambda x: embed_aggregate(x, embeds, **pooling[p]))
        train_df.to_csv(os.path.join(output_dir, 'train_clear.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_clear.csv'), index=False)
    else:
        for col in train_df.columns:
            if col.startswith('comment_seq'):
                train_df[col] = train_df[col].apply(lambda x: parse_seq(x, int))
                test_df[col] = test_df[col].apply(lambda x: parse_seq(x, int))
            elif col.startswith('comment_vec'):
                train_df[col] = train_df[col].apply(lambda x: parse_seq(x, float))
                test_df[col] = test_df[col].apply(lambda x: parse_seq(x, float))


    logger.debug('Embedding matrix shape = {}'.format(embeds.shape))
    logger.debug('Number of null word embeddings = {}'.format(np.sum(np.sum(embeds.matrix, axis=1) == 0)))

    # ====Train/test split data====
    x = np.array(train_df[seq_col_name].values.tolist())
    y = np.array(train_df[target_labels].values.tolist())
    x_train_nn, x_val_nn, y_train, y_val, train_idxs, val_idxs = split_data(x, y, test_size=0.2, shuffle=True, random_state=42)
    x_test_nn = np.array(test_df[seq_col_name].values.tolist())

    x_train_tfidf = train_df['comment_text_clear'].values[train_idxs]
    x_val_tfidf = train_df['comment_text_clear'].values[val_idxs]
    x_test_tfidf = test_df['comment_text_clear'].values

    catboost_cols = catboost_features(train_df, test_df)
    x_train_cb = train_df[catboost_cols].values[train_idxs].T
    x_val_cb = train_df[catboost_cols].values[val_idxs].T
    x_test_cb = test_df[catboost_cols].values.T

    # ====Train models====
    nn_models = {
        'cnn': cnn,
        'dense': dense,
        'rnn': rnn
    }

    params = Params(config)

    metrics = {}
    predictions = {}
    for param in params['models']:
        for model_label, model_params in param.items():
            if model_params.get('common', {}).get('warm_start', False) and os.path.exists(model_params.get('common', {}).get('model_file', '')):
                logger.info('{} warm starting...'.format(model_label))
                model = load_model(model_params.get('common', {}).get('model_file', None))
            elif model_label in nn_models:
                model = nn_models[model_label](
                            embeds.matrix,
                            num_classes,
                            max_seq_len,
                            gpus=gpus,
                            **model_params['init'])
                model_alias = model_params.get('common', {}).get('alias', None)
                if model_alias is None or not model_alias:
                    model_alias = '{}_{}'.format(model_label, i)
                logger.info("training {} ...".format(model_label))
                hist = train(x_train_nn,
                             y_train,
                             model,
                             logger=logger,
                             **model_params['train'])
                predictions[model_alias] = model.predict(x_val_nn)
                save_predictions(test_df, model.predict(x_test_nn), target_labels, model_alias)
            elif model_label == 'tfidf':
                model = TFIDF(target_labels, **model_params['init'])
                model.fit(x_train_tfidf, y_train, **model_params['train'])
                predictions[model_alias] = model.predict(x_val_tfidf)
                save_predictions(test_df, model.predict(x_test_tfidf), target_labels, model_alias)
            elif model_label == 'catboost':
                model = CatBoost(target_labels, **model_params['init'])
                model.fit(x_train_cb, y_train, eval_set=(x_val_cb, y_val), use_best_model=True)
                predictions[model_alias] = model.predict_proba(x_val_cb)
                save_predictions(test_df, model.predict_proba(x_test_cb), target_labels, model_alias)
            metrics[model_alias] = get_metrics(y_val, predictions[model_alias], target_labels)
            logger.debug('{} params:\n{}'.format(model_alias, model_params))
            logger.debug('{} metrics:\n{}'.format(model_alias, print_metrics(metrics[model_alias])))
            #model.save(model_file[model_label])
            #joblib.dump(model, model_file['lr'].format(label))

    logger.info('Saving metrics...')
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        f.write(json.dumps(metrics))

    # ====END OF `VALIDATE`====
    if mode == 'validate':
        return True

    # Meta catboost
    logger.info('training catboost as metamodel...')

    x_meta = [predictions[model_alias] for model_alias in sorted(predictions.keys())]
    x_meta = np.array(x_train_meta).T

    x_train_meta, x_val_meta, y_train_meta, y_val_meta = train_test_split(x_meta, y_val, test_size=0.20, random_state=42)
    meta_model = CatBoost(target_labels,
                         loss_function='Logloss',
                         iterations=1000,
                         depth=6,
                         learning_rate=0.03,
                         rsm=1
    )
    meta_model.fit(x_train_meta, y_train_meta, eval_set=(x_val_meta, y_val_meta), use_best_model=True)
    y_hat_meta = meta_model.predict_proba(x_val_meta)
    metrics_meta = get_metrics(y_val_meta, y_hat_meta, target_labels)
    #joblib.dump(model, model_file['catboost'].format(label))
    logger.debug('{} metrics:\n{}'.format('META', print_metrics(metrics_meta)))

    # ====Predict====
    logger.info('Applying models...')
    test_cols = []
    for model_alias in sorted(predictions.keys()):
        for label in target_labels:
            test_cols.append('{}_{}'.format(model_alias, label))
    x_test = test_df[test_cols].values

    preds = meta_model.predict_proba(x_test)
    for i, label in enumerate(target_labels):
        test_df[label] = preds[:, i]

    # ====Normalize probabilities====
    if norm_prob:
        for label in target_labels:
            test_df[label] = norm_prob_koef * test_df[label]

    # ====Save results====
    logger.info('Saving results...')
    test_df[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].to_csv(result_fname, index=False, header=True)
    test_df.to_csv('{}_tmp'.format(result_fname), index=False, header=True)


if __name__=='__main__':
    main()
