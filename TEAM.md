# ML Team — Legal Text Classification

## Member 1 — Saifulla: Dataset, Preprocessing & TF-IDF/LDA
**Branch:** `member1-tfidf-lda`

- Load the Supreme Court dataset; inspect record counts, missing labels, and class distribution.
- Clean text and prepare labels; establish the shared reproducible train/test split.
- Implement TF-IDF feature extraction (fit on training text only).
- Implement LDA + Logistic Regression baseline.
- Implement class-weighted Logistic Regression variant.
- Report model metrics and document preprocessing/methodology.
- Commit work to this branch and open a pull request.

**Deliverables:** dataset/preprocessing module, TF-IDF/LDA model module, two baseline models, and methodology notes.

## Member 2 — Teammate: Doc2Vec, Evaluation & Demo
**Branch:** `member2-doc2vec`

- Implement Doc2Vec embeddings trained on training documents only; infer vectors for test text.
- Train Logistic Regression on Doc2Vec vectors.
- Evaluate with accuracy, precision, recall, weighted F1, and confusion matrix.
- Compare all models using the same split and metrics.
- Implement prediction/demo module and maintain the Streamlit demo.
- Commit work to this branch and open a pull request.

**Deliverables:** Doc2Vec model module, evaluation/comparison module, prediction/demo, and results.

## Shared — Integration & Review
- Use the same dataset filtering, cleaned text, labels, random seed, and train/test split for fair comparison.
- Fit all representations and models using training data only; avoid data leakage.
- Review each other's pull requests and resolve conflicts together.
- Run the full pipeline and demo before review.
- Finalize README, requirements, report, presentation, and viva preparation.
- Distinguish notebook-recorded scores from results obtained by the cleaned implementation.

## Git workflow
1. Work only on your assigned branch.
2. Make small, descriptive commits.
3. Push branch and open a pull request into `main`.
4. Review the other member's PR; merge only after testing.
5. Do not commit downloaded datasets, model binaries, secrets, or large generated files.

## Current files
- `legal_text_classifier.py` — existing training pipeline (legacy entry point).
- `app.py` — Streamlit demo.
- `requirements.txt` — dependencies.
- `README.md` — setup, run, and notebook results.
- `Legal_Text_Classification_Writeup.pdf` — project write-up.
- `src/` — new modular model implementation (team branches).
