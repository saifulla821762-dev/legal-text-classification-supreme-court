"""Leakage-aware Doc2Vec embedding and Logistic Regression model."""
from __future__ import annotations

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression


def fit_doc2vec(X_train_text: list[str], X_test_text: list[str],
                y_train: list[int], vector_size: int = 50,
                epochs: int = 20, random_state: int = 42):
    """Train Doc2Vec on training documents only; infer test embeddings."""
    tagged_train = [
        TaggedDocument(words=text.split(), tags=[index])
        for index, text in enumerate(X_train_text)
    ]
    embedding = Doc2Vec(
        vector_size=vector_size, window=5, min_count=2, workers=2,
        epochs=epochs, seed=random_state,
    )
    embedding.build_vocab(tagged_train)
    embedding.train(tagged_train, total_examples=embedding.corpus_count,
                   epochs=embedding.epochs)
    train_vectors = np.asarray([embedding.dv[i] for i in range(len(X_train_text))])
    test_vectors = np.asarray([
        embedding.infer_vector(text.split(), epochs=epochs) for text in X_test_text
    ])
    classifier = LogisticRegression(max_iter=1000, random_state=random_state)
    classifier.fit(train_vectors, y_train)
    predictions = classifier.predict(test_vectors)
    return embedding, classifier, train_vectors, test_vectors, predictions


def infer_doc2vec(embedding: Doc2Vec, text: str, epochs: int = 20):
    """Infer a vector for one cleaned text sample."""
    return embedding.infer_vector(text.split(), epochs=epochs)
