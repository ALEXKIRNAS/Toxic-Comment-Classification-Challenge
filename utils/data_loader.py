import os
import pickle
import re
import string

import pandas as pd
from keras.preprocessing import text, sequence
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.constants import CLASS_NAMES, MAX_FEATURES, MAX_LEN
from utils.preprocessing import clean_text, get_embedingsget_embeddings


def load_data(train_df_path, test_df_path, embedings_file):
    train = pd.read_csv(train_df_path).fillna(' ')
    test = pd.read_csv(test_df_path).fillna(' ')

    submission = pd.DataFrame.from_dict({'id': test['id']})
    train_submission = pd.DataFrame.from_dict({'id': train['id']})

    targets_train = train[CLASS_NAMES].values
    train = train["comment_text"].map(clean_text).values
    test = test["comment_text"].map(clean_text).values

    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(train) + list(test))

    X_train = tokenizer.texts_to_sequences(train)
    X_test = tokenizer.texts_to_sequences(test)

    x_train = sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)

    embedings = get_embedingsget_embeddings(tokenizer, embedings_file)

    return x_train, targets_train, x_test, train_submission, submission, embedings


def tokenize(s):
    re_tok = re.compile('([%s“”¨«»®´·º½¾¿¡§£₤‘’])' % string.punctuation)
    return re_tok.sub(r' \1 ', clean_text(s)).split()


def tf_idf_vectors(train_df, test_df, preprocess):
    train_text = train_df['comment_text']
    test_text = test_df['comment_text']
    all_text = pd.concat([train_text, test_text])

    if not os.path.exists('./train_tfidf_features.pkl') or not os.path.exists('./test_tfidf_features.pkl'):
        if preprocess:
            word_vectorizer = TfidfVectorizer(analyzer='word',
                                              ngram_range=(1, 2),
                                              tokenizer=tokenize,
                                              max_df=0.9,
                                              min_df=3,
                                              strip_accents='unicode',
                                              use_idf=True,
                                              smooth_idf=True,
                                              sublinear_tf=True,
                                              max_features=300000)

            char_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                              smooth_idf=True,
                                              tokenizer=tokenize,
                                              strip_accents='unicode',
                                              analyzer='char',
                                              max_df=0.9,
                                              min_df=3,
                                              ngram_range=(1, 4),
                                              max_features=300000)
        else:
            word_vectorizer = TfidfVectorizer(analyzer='word',
                                              ngram_range=(1, 2),
                                              max_df=0.9,
                                              min_df=3,
                                              strip_accents='unicode',
                                              use_idf=True,
                                              smooth_idf=True,
                                              sublinear_tf=True,
                                              max_features=300000)

            char_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                              smooth_idf=True,
                                              strip_accents='unicode',
                                              analyzer='char',
                                              max_df=0.9,
                                              min_df=3,
                                              ngram_range=(1, 4),
                                              max_features=300000)

        word_vectorizer.fit(all_text)
        train_word_features = word_vectorizer.transform(train_text)
        test_word_features = word_vectorizer.transform(test_text)

        char_vectorizer.fit(all_text)
        train_char_features = char_vectorizer.transform(train_text)
        test_char_features = char_vectorizer.transform(test_text)

        train_word_features = sparse.hstack([train_char_features, train_word_features])
        test_word_features = sparse.hstack([test_char_features, test_word_features])

        with open('./train_tfidf_features.pkl', 'wb') as f:
            pickle.dump(train_word_features, f)
        with open('./test_tfidf_features.pkl', 'wb') as f:
            pickle.dump(test_word_features, f)
    else:
        with open('train_tfidf_features.pkl', 'rb') as f:
            train_word_features = pickle.load(f)
        with open('test_tfidf_features.pkl', 'rb') as f:
            test_word_features = pickle.load(f)

    return train_word_features, test_word_features
