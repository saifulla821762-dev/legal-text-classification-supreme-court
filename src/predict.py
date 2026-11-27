"""Prediction helpers and Supreme Court issue-area display labels."""
from __future__ import annotations

import re\n\ndef clean_text(text: str) -> str:\n    text = text.lower()\n    text = re.sub(r"\\s+", " ", text)\n    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)\n    return text.strip()

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
