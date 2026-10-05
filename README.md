# Legal Text Classification of U.S. Supreme Court Decisions

This project classifies U.S. Supreme Court decision text into its high-level legal issue area. The supplied Colab notebook compared TF-IDF + LDA + Logistic Regression, class-weighted Logistic Regression, and Doc2Vec + Logistic Regression.

## Dataset

The project uses the Supreme Court decisions collection exposed by `textacy.datasets.SupremeCourt`. It contains about 8,400 modern-era decisions (November 1946 through June 2016), with decision text and issue-area metadata. The notebook loaded 8,419 records and retained 8,396 after excluding records with missing issue or issue-area labels. The corpus is downloaded automatically by `textacy` the first time the script runs.

Dataset documentation: https://textacy.readthedocs.io/en/0.11.0/api_reference/datasets.html

## Setup

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python legal_text_classifier.py
```

The script downloads and loads the corpus, filters and cleans records, trains three classifiers, prints accuracy and classification reports, and predicts the issue area for a sample legal passage. Initial corpus download requires internet access. Training can take a few minutes on a CPU.

## Results recorded in the supplied Colab notebook

| Model | Accuracy | Weighted F1 |
|---|---:|---:|
| TF-IDF + LDA + Logistic Regression | 44.82% | 0.36 |
| Class-weighted TF-IDF + LDA + Logistic Regression | 23.87% | 0.22 |
| Doc2Vec + Logistic Regression | 66.49% | 0.66 |

These are results recorded in the notebook, not scores from running this cleaned script. The notebook trained Doc2Vec on all documents before splitting vectors into train and test sets. Test-document text therefore affected representation learning and may inflate the apparent holdout score. This script splits first and learns Doc2Vec from training documents only; its measured scores may differ. The script computes metrics directly instead of hard-coding summary F1 values as the notebook does.

## Files

- `legal_text_classifier.py` - training and sample-prediction pipeline.
- `requirements.txt` - Python dependencies.
- [Legal_Text_Classification_Writeup.pdf](Legal_Text_Classification_Writeup.pdf) - two-page project summary.

## Limitations

The classes are imbalanced; one issue area has only one example in the notebook's filtered corpus. The notebook's random split was not stratified. This is an educational text-classification demonstration, not a legal decision tool. Scores depend on corpus and package versions and Doc2Vec training settings.