#!/usr/bin/env python3
"""
CBOW module
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    creates a bag of words embedding matrix:
    - sentences: list of sentences to analyze
    - vocab: list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used
    Returns: embeddings, features
        - embeddings: numpy.ndarray of shape (s, f) containing
        the embeddings
            - s: number of sentences in sentences
            - f: number of features analyzed
        features is a list of the features used for embeddings
    """
    vector = CountVectorizer(vocabulary=vocab)
    X = vector.fit_transform(sentences)
    features = vector.get_feature_names()
    embeddings = X.toarray()

    return embeddings, features
