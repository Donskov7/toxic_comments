import numpy as np


def calc_text_uniq_words(text):
    unique_words = set()
    for word in text.split():
        unique_words.add(word)
    return len(unique_words)


def catboost_features(train_df, test_df):
    train_df['text_unique_len'] = train_df['comment_text_clear'].apply(calc_text_uniq_words)
    test_df['text_unique_len'] = test_df['comment_text_clear'].apply(calc_text_uniq_words)

    train_df['text_unique_koef'] = train_df['text_unique_len'] / train_df['text_len']
    test_df['text_unique_koef'] = test_df['text_unique_len'] / test_df['text_len']

    return ['text_len', 'text_unique_len', 'text_unique_koef', 'comment_vec_max', 'comment_vec_avg']


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
