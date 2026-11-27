"""Streamlit demo for Supreme Court legal issue-area classification."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
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
    text = text.lower()
    text = re.sub(r"\\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
    return text.strip()

@st.cache_resource(show_spinner="Downloading corpus and training the classifiers…")
def train_all_models():
    dataset = SupremeCourt()
    dataset.download()
    texts, labels = [], []
    for text, metadata in dataset.records():
        label = metadata.get("issue_area")
        issue = metadata.get("issue")
        if label in (None, -1) or issue in (None, "none"):
            continue
        texts.append(clean_text(text))
        labels.append(int(label))

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    train_tfidf = vectorizer.fit_transform(X_train)
    test_tfidf = vectorizer.transform(X_test)
    n_topics = min(14, len(set(y_train)))
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    train_topics = lda.fit_transform(train_tfidf)
    test_topics = lda.transform(test_tfidf)

    models = {}
    metrics = []
    confusions = {}
    for name, model in [
        ("TF-IDF + LDA + Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Class-weighted TF-IDF + LDA + Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ]:
        model.fit(train_topics, y_train)
        preds = model.predict(test_topics)
        p, r, f, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
        metrics.append({"Model": name, "Accuracy": accuracy_score(y_test, preds), "Precision (weighted)": p, "Recall (weighted)": r, "F1 (weighted)": f})
        confusions[name] = confusion_matrix(y_test, preds, labels=sorted(set(labels)))
        models[name] = model

    tagged = [TaggedDocument(words=t.split(), tags=[i]) for i, t in enumerate(X_train)]
    d2v = Doc2Vec(vector_size=50, window=5, min_count=2, workers=2, epochs=20, seed=42)
    d2v.build_vocab(tagged)
    d2v.train(tagged, total_examples=d2v.corpus_count, epochs=d2v.epochs)
    train_vecs = np.asarray([d2v.dv[i] for i in range(len(X_train))])
    test_vecs = np.asarray([d2v.infer_vector(t.split(), epochs=20) for t in X_test])
    d2v_lr = LogisticRegression(max_iter=1000, random_state=42)
    d2v_lr.fit(train_vecs, y_train)
    d2v_preds = d2v_lr.predict(test_vecs)
    p, r, f, _ = precision_recall_fscore_support(y_test, d2v_preds, average="weighted", zero_division=0)
    name = "Doc2Vec + Logistic Regression"
    metrics.append({"Model": name, "Accuracy": accuracy_score(y_test, d2v_preds), "Precision (weighted)": p, "Recall (weighted)": r, "F1 (weighted)": f})
    confusions[name] = confusion_matrix(y_test, d2v_preds, labels=sorted(set(labels)))
    models[name] = d2v_lr
    return vectorizer, lda, d2v, models, metrics, confusions, sorted(set(labels))

st.set_page_config(page_title="Legal Text Classification", page_icon="⚖️", layout="wide")
st.title("Legal Text Classification")
st.caption("Predict a broad issue-area category for U.S. Supreme Court decision text.")
st.info("The first run downloads the dataset and trains the models. This may take several minutes and requires an internet connection.")

with st.spinner("Preparing dataset and models…"):
    vectorizer, lda, d2v, models, metrics, confusions, class_labels = train_all_models()

st.subheader("Model evaluation")
metric_df = pd.DataFrame(metrics).set_index("Model")
st.dataframe(metric_df.style.format("{:.3f}"), use_container_width=True)

selected_metric_model = st.selectbox("Confusion matrix for", list(confusions.keys()))
cm = confusions[selected_metric_model]
labels = [ISSUE_AREA_NAMES.get(c, str(c)) for c in class_labels]
st.dataframe(pd.DataFrame(cm, index=labels, columns=labels), use_container_width=True)

st.subheader("Classify legal text")
input_text = st.text_area("Paste a legal passage or case excerpt", height=220, placeholder="Enter legal text here…")
model_choice = st.selectbox("Prediction model", list(models.keys()), index=2)
if st.button("Predict issue area", type="primary", disabled=not input_text.strip()):
    cleaned = clean_text(input_text)
    if model_choice == "Doc2Vec + Logistic Regression":
        features = d2v.infer_vector(cleaned.split(), epochs=20).reshape(1, -1)
    else:
        features = lda.transform(vectorizer.transform([cleaned]))
    pred = int(models[model_choice].predict(features)[0])
    st.success(f"Predicted category: **{ISSUE_AREA_NAMES.get(pred, 'Unknown')}** (issue area {pred})")

st.caption("Educational classification prototype. Predictions are categories, not legal advice.")
