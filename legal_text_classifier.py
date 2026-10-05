"""Train issue-area classifiers for U.S. Supreme Court decision text."""

from __future__ import annotations

import re

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from textacy.datasets import SupremeCourt


ISSUE_AREA_NAMES = {
    1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment",
    4: "Due Process", 5: "Privacy", 6: "Attorneys", 7: "Unions",
    8: "Economic Activity", 9: "Judicial Power", 10: "Federalism",
    11: "Interstate Relations", 12: "Federal Taxation",
    13: "Miscellaneous", 14: "Private Action",
}


def clean_text(text: str) -> str:
    """Lowercase text and normalize punctuation and whitespace."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


def load_records() -> tuple[list[str], list[int]]:
    """Download/load textacy's corpus and keep records with usable labels."""
    dataset = SupremeCourt()
    dataset.download()
    records = list(dataset.records())
    texts: list[str] = []
    labels: list[int] = []
    for text, metadata in records:
        issue = metadata.get("issue")
        issue_area = metadata.get("issue_area")
        if issue_area in (None, -1) or issue in (None, "none"):
            continue
        texts.append(clean_text(text))
        labels.append(int(issue_area))
    print(f"Loaded records: {len(records)}")
    print(f"Usable records: {len(texts)}")
    print(f"Issue areas: {len(set(labels))}")
    return texts, labels


def show_report(name: str, y_true: list[int], y_pred: np.ndarray) -> None:
    """Print computed accuracy and per-class scores."""
    print(f"\n{name}\n{'=' * len(name)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))


def main() -> None:
    texts, labels = load_records()
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    # Learn TF-IDF vocabulary and IDF weights from training documents only.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    topic_count = min(14, len(set(y_train)))
    lda = LatentDirichletAllocation(
        n_components=topic_count, random_state=42, learning_method="batch"
    )
    X_train_topics = lda.fit_transform(X_train_tfidf)
    X_test_topics = lda.transform(X_test_tfidf)

    baseline = LogisticRegression(max_iter=1000, random_state=42)
    baseline.fit(X_train_topics, y_train)
    show_report("TF-IDF + LDA + Logistic Regression", y_test,
                baseline.predict(X_test_topics))

    balanced = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    balanced.fit(X_train_topics, y_train)
    show_report("Class-weighted TF-IDF + LDA + Logistic Regression", y_test,
                balanced.predict(X_test_topics))

    # Train Doc2Vec only on training documents, then infer vectors for test text.
    tagged_train = [
        TaggedDocument(words=text.split(), tags=[i])
        for i, text in enumerate(X_train_text)
    ]
    doc2vec = Doc2Vec(
        vector_size=50, window=5, min_count=2, workers=2, epochs=20, seed=42
    )
    doc2vec.build_vocab(tagged_train)
    doc2vec.train(tagged_train, total_examples=doc2vec.corpus_count,
                  epochs=doc2vec.epochs)
    X_train_doc2vec = np.asarray(
        [doc2vec.dv[i] for i in range(len(X_train_text))]
    )
    X_test_doc2vec = np.asarray([
        doc2vec.infer_vector(text.split(), epochs=20) for text in X_test_text
    ])
    doc2vec_lr = LogisticRegression(max_iter=1000, random_state=42)
    doc2vec_lr.fit(X_train_doc2vec, y_train)
    show_report("Doc2Vec + Logistic Regression", y_test,
                doc2vec_lr.predict(X_test_doc2vec))

    sample = (
        "The Supreme Court considered whether the government violated the "
        "constitutional rights of the defendant during criminal proceedings."
    )
    sample_vector = doc2vec.infer_vector(clean_text(sample).split(), epochs=20)
    predicted_class = int(doc2vec_lr.predict([sample_vector])[0])
    print("\nSample prediction")
    print(f"Issue area: {ISSUE_AREA_NAMES.get(predicted_class, 'Unknown')}")
    print(f"Class: {predicted_class}")


if __name__ == "__main__":
    main()
