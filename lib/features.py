from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def calc_text_uniq_words(text):
    unique_words = set()
    for word in text.split():
        unique_words.add(word)
    return len(unique_words)


def get_tfidf(x_train, x_val, x_test, max_features=50000):
    word_tfidf = TfidfVectorizer(max_features=max_features, analyzer='word', lowercase=True, ngram_range=(1, 3), token_pattern='[a-zA-Z0-9]')
    char_tfidf = TfidfVectorizer(max_features=max_features, analyzer='char', lowercase=True, ngram_range=(1, 5), token_pattern='[a-zA-Z0-9]')

    train_tfidf_word = word_tfidf.fit_transform(x_train)
    val_tfidf_word = word_tfidf.transform(x_val)
    test_tfidf_word = word_tfidf.transform(x_test)

    train_tfidf_char = char_tfidf.fit_transform(x_train)
    val_tfidf_char = char_tfidf.transform(x_val)
    test_tfidf_char = char_tfidf.transform(x_test)

    train_tfidf = sparse.hstack([train_tfidf_word, train_tfidf_char])
    val_tfidf = sparse.hstack([val_tfidf_word, val_tfidf_char])
    test_tfidf = sparse.hstack([test_tfidf_word, test_tfidf_char])

    return train_tfidf, val_tfidf, test_tfidf, word_tfidf, char_tfidf


def get_most_informative_features(vectorizers, clf, n=20):
    feature_names = []
    for vectorizer in vectorizers:
        feature_names.extend(vectorizer.get_feature_names())
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    return coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1]


def get_bow(texts, words):
    result = np.zeros((len(texts), len(words)))
    print(np.shape(result))
    for i, text in tqdm(enumerate(texts)):
        for j, word in enumerate(words):
            try:
                if word in text:
                    result[i][j] = 1
            except UnicodeDecodeError:
                pass
    return result
