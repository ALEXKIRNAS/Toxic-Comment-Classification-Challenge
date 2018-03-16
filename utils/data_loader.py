import pandas as pd

from keras.preprocessing import text, sequence
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
