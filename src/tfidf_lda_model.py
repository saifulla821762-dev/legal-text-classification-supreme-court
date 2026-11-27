"""TF-IDF/LDA feature pipeline and Logistic Regression baselines."""
from __future__ import annotations

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def fit_tfidf_lda(X_train_text: list[str], X_test_text: list[str],
                   y_train: list[int], n_topics: int = 14,
                   random_state: int = 42):
    """Fit text features and LDA using training text only."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    train_tfidf = vectorizer.fit_transform(X_train_text)
    test_tfidf = vectorizer.transform(X_test_text)

    topic_count = min(n_topics, len(set(y_train)))
    lda = LatentDirichletAllocation(
        n_components=topic_count, random_state=random_state, learning_method="batch"
    )
    train_topics = lda.fit_transform(train_tfidf)
    test_topics = lda.transform(test_tfidf)

    models = {
        "TF-IDF + LDA + Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "Class-weighted TF-IDF + LDA + Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
    }
    fitted = {}
    predictions = {}
    for name, model in models.items():
        model.fit(train_topics, y_train)
        fitted[name] = model
        predictions[name] = model.predict(test_topics)
    return vectorizer, lda, fitted, predictions
