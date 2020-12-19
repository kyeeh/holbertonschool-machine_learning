#!/usr/bin/env python3
"""
TF-IDF module
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    creates a TF-IDF embedding:
    - sentences: list of sentences to analyze
    - vocab: list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used
    Returns: embeddings, features
        - embeddings is a numpy.ndarray of shape (s, f) containing
        the embeddings
            - s: number of sentences in sentences
            - f: number of features analyzed
        features is a list of the features used for embeddings
    """
    tfidf = TfidfVectorizer(vocabulary=vocab)
    X = tfidf.fit_transform(sentences)
    features = tfidf.get_feature_names()
    embeddings = X.toarray()

    return embeddings, features
