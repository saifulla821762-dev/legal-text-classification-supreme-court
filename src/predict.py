"""Prediction helpers and Supreme Court issue-area display labels."""
from __future__ import annotations

from .preprocessing import clean_text

ISSUE_AREA_NAMES = {
    1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment",
    4: "Due Process", 5: "Privacy", 6: "Attorneys", 7: "Unions",
    8: "Economic Activity", 9: "Judicial Power", 10: "Federalism",
    11: "Interstate Relations", 12: "Federal Taxation",
    13: "Miscellaneous", 14: "Private Action",
}


def predict_doc2vec(text: str, embedding, classifier, epochs: int = 20):
    """Predict the issue-area label for a single passage using Doc2Vec."""
    cleaned = clean_text(text)
    vector = embedding.infer_vector(cleaned.split(), epochs=epochs)
    predicted = int(classifier.predict([vector])[0])
    return predicted, ISSUE_AREA_NAMES.get(predicted, "Unknown")
