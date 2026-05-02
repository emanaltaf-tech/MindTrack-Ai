# Adaptive AI-Based Study Productivity System (MindTrack AI)

A machine learning system that analyses study session behaviour and predicts
focus scores using three ML paradigms: Regression, Classification, and Clustering.

## Tech Stack
- scikit-learn (RandomForest Regression, Classification, MiniBatchKMeans Clustering)
- Gradio (interactive web UI)
- matplotlib (visualisations)
- pandas / numpy

## Setup & Run

```bash
pip install -r requirements.txt
python app.py
# Then open http://localhost:7860
```

## Using Real Data (Recommended)

1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset

2. Rename the downloaded CSV to `kaggle_raw.csv`

3. Place `kaggle_raw.csv` in this folder (same level as app.py)

4. Delete `dataset.csv` if it exists (it will be regenerated automatically)

5. Run `python app.py` — the app will auto-detect and use real data

## Dataset Schema

Both real and synthetic modes produce the same columns:

| Column | Description |
|--------|-------------|
| study_time | Minutes of active focused study |
| break_time | Minutes of intentional breaks |
| tasks_completed | Number of tasks/problems finished |
| idle_time | Minutes of distraction/idle time |
| session_time | Total session duration (minutes) |
| time_of_day | morning / afternoon / evening / night |
| focus_score | 0–100 target score (regression target) |

## ML Models

- **RandomForest Regressor** — predicts focus_score (0-100)
- **RandomForest Classifier** — predicts productive / non-productive (threshold: 60)
- **K-Means Clustering (k=3-5)** — groups students into behaviour patterns; k chosen by silhouette score
- **5-Fold Cross-Validation** — honest performance estimate with per-fold breakdown shown in the UI

## Feature Engineering

Three derived features are computed before training to capture session *ratios*
that are more informative than raw minute counts:

| Feature | Formula | Meaning |
|---------|---------|---------|
| `study_efficiency` | study_time / session_time | Fraction of session spent in active study |
| `idle_ratio` | idle_time / session_time | Distraction load relative to session length |
| `break_ratio` | break_time / study_time | Recovery rate relative to study effort |

## Preprocessing Notes

- **Dummy variable trap avoidance**: `drop_first=True` in one-hot encoding removes one redundant dummy column (morning), keeping the feature matrix full-rank.
- **No data leakage**: StandardScaler is fitted exclusively on the training split and applied to test/inference.
- **Stratified split**: `train_test_split` uses `stratify=y_cls` to preserve the class balance in both splits.

## Project Structure

```
study-productivity-ai/
├── app.py               # Main Gradio application
├── generate_dataset.py  # Data loading (real) or generation (synthetic)
├── models.py            # ML model training + evaluation
├── preprocessing.py     # Feature engineering, encoding + scaling
├── recommendations.py   # Rule-based recommendation engine
├── charts.py            # Matplotlib chart builders (incl. CV folds, silhouette charts)
├── requirements.txt
├── README.md
├── kaggle_raw.csv       # ← Add this (download from Kaggle)
└── dataset.csv          # Auto-generated on first run
```
