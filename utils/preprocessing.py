import re
import numpy as np

from utils.constants import  MAX_FEATURES, EMBBEDINGS_SIZE


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)


    # text = re.sub(r'(.)\1{2,}', r'\1', text)
    # text = re.sub(r"fuck", ' fuck ', text)
    # text = re.sub(r"idiot", ' idiot ', text)
    # text = re.sub(r"stupid", ' stupid ', text)
    # text = re.sub(r"dick", ' dick ', text)
    # text = re.sub(r"shit", ' shit ', text)

    # text = re.sub(r"US", "United States", text)
    # text = re.sub(r"IT", "Information Technology", text)
    # text = re.sub(r"(W|w)on\'t", "will not", text)
    # text = re.sub(r"(C|c)an\'t", "can not", text)
    # text = re.sub(r"(I|i)\'m", "i am", text)
    # text = re.sub(r"(A|a)in\'t", "is not", text)
    # text = re.sub(r"(\w+)\'ll", "\g<1> will", text)
    # text = re.sub(r"(\w+)n\'t", "\g<1> not", text)
    # text = re.sub(r"(\w+)\'ve", "\g<1> have", text)
    # text = re.sub(r"(\w+)\'s", "\g<1> is", text)
    # text = re.sub(r"(\w+)\'re", "\g<1> are", text)
    # text = re.sub(r"(\w+)\'d", "\g<1> would", text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+$', r'', text)
    text = re.sub(r'[\n\t\b\r]', '', text)
    text = text.strip(' ')

    return text


def _get_emb_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def _get_embedings_indexes(embedings_path):
    embeddings_index = dict(_get_emb_coefs(*o.rstrip().rsplit(' '))
                            for o in open(embedings_path, encoding='utf-8'))

    return embeddings_index


def get_embedingsget_embeddings(tokenizer, embedings_path):
    embeddings_index = _get_embedings_indexes(embedings_path)

    word_index = tokenizer.word_index
    nb_words = min(MAX_FEATURES, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBBEDINGS_SIZE))

    for word, i in word_index.items():
        if i >= MAX_FEATURES: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
