# ML Team — Legal Text Classification

## Member 1 — Saifulla: Dataset, Preprocessing & TF-IDF/LDA Models
**Branch:** `member1-data-tfidf`

- Load the Supreme Court dataset and verify record counts, labels, missing values, and class distribution.
- Implement and document text cleaning and train/test split.
- Build TF-IDF features (fit on training data only).
- Implement LDA + Logistic Regression baseline.
- Implement class-weighted Logistic Regression variant.
- Evaluate both models with accuracy, precision, recall, weighted F1, and confusion matrix.
- Maintain the dataset/preprocessing/methodology sections of the report and README.
- Commit work to the assigned branch and open a pull request.

**Deliverables:** reproducible preprocessing, two TF-IDF/LDA models, their evaluation outputs, and methodology notes.

## Member 2 — Teammate: Doc2Vec, Evaluation & Demo
**Branch:** `member2-doc2vec-evaluation`

- Implement Doc2Vec training using training documents only; infer vectors for test documents.
- Train Logistic Regression on Doc2Vec embeddings.
- Evaluate Doc2Vec with accuracy, precision, recall, weighted F1, and confusion matrix.
- Compare Doc2Vec results against Member 1's models using the same split and metrics.
- Maintain/validate the Streamlit demo (`app.py`) for model selection and pasted-text prediction.
- Maintain results/comparison/demo sections of the report and README.
- Commit work to the assigned branch and open a pull request.

**Deliverables:** leakage-aware Doc2Vec model, comparison results, and a tested prediction demo.

## Shared — Integration & Review
- Use the same dataset filtering, cleaned text, labels, random seed, and train/test split across all models.
- Check for data leakage; fit every representation/model using training data only.
- Review one another's pull requests; resolve conflicts together.
- Run the complete pipeline and demo before the review.
- Finalize README, requirements, two-page report, presentation, and viva preparation.
- Clearly distinguish notebook-recorded scores from scores produced by the cleaned implementation.

## Git Workflow
1. Work only on your assigned branch.
2. Commit small, descriptive changes (for example, `feat: add TF-IDF baseline` or `feat: train Doc2Vec model`).
3. Push the branch and open a pull request into `main`.
4. The other member reviews; merge after testing.
5. Do not commit downloaded datasets, model binaries, secrets, or large generated files.

## Current Repository Files
- `legal_text_classifier.py` — training pipeline
- `app.py` — Streamlit demo
- `requirements.txt` — dependencies
- `README.md` — setup, run, and recorded notebook results
- `Legal_Text_Classification_Writeup.pdf` — project write-up
