# MindTrack AI — Adaptive Study Productivity System

A machine learning system that analyses study session behaviour and predicts focus scores using three ML paradigms: Regression, Classification, and Clustering.

---

## Tech Stack

- **scikit-learn** — RandomForest Regression, Classification, MiniBatchKMeans
- **Gradio** — Interactive web UI
- **matplotlib** — Visualisations
- **pandas / numpy** — Data handling

---

## Setup & Run

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:7860](http://localhost:7860)

---

## Using Real Data (Recommended)

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset

2. Rename the file to `kaggle_raw.csv` and place it in the project folder

3. Delete `dataset.csv` if it exists — it will be regenerated automatically

4. Run `python app.py` — the app auto-detects and uses real data

If `kaggle_raw.csv` is not found, the app falls back to a synthetic dataset generated automatically on first run.

---

## Project Structure

```
study-productivity-ai/
├── app.py               # Main Gradio application (4-screen flow)
├── generate_dataset.py  # Data loading (real) or generation (synthetic)
├── models.py            # RF Regressor, RF Classifier, KMeans training & evaluation
├── preprocessing.py     # Feature engineering, encoding, scaling
├── recommendations.py   # Rule-based recommendation engine
├── charts.py            # matplotlib chart builders
├── requirements.txt
├── README.md
├── kaggle_raw.csv       # ← Add this (download from Kaggle)
└── dataset.csv          # Auto-generated on first run
```

---

## ML Models

| Model | Algorithm | Output |
|-------|-----------|--------|
| Regression | RandomForestRegressor | Focus score (0–100) |
| Classification | RandomForestClassifier | Productive / Non-Productive |
| Clustering | MiniBatchKMeans (k=3–5) | Behaviour pattern group |

Cluster k is selected automatically using silhouette score. 5-fold cross-validation is used for honest performance estimates.

---

## Dataset Schema

| Column | Description |
|--------|-------------|
| `study_time` | Minutes of active focused study |
| `break_time` | Minutes of intentional breaks |
| `tasks_completed` | Number of tasks/problems finished |
| `idle_time` | Minutes of distraction/idle time |
| `session_time` | Total session duration (minutes) |
| `time_of_day` | morning / afternoon / evening / night |
| `focus_score` | 0–100 target score (regression target) |

---

## Feature Engineering

Three derived features are computed before training:

| Feature | Formula | Meaning |
|---------|---------|---------|
| `study_efficiency` | study_time / session_time | Fraction of session in active study |
| `idle_ratio` | idle_time / session_time | Distraction load relative to session |
| `break_ratio` | break_time / study_time | Recovery rate relative to study effort |

---

## Preprocessing Notes

- `drop_first=True` in one-hot encoding avoids the dummy variable trap
- `StandardScaler` is fitted on training data only — no data leakage
- `train_test_split` uses `stratify=y_cls` to preserve class balance

---

## Learner Types (Clusters)

| Type | Description |
|------|-------------|
|  Deep Focus Worker | High study efficiency, low idle time |
|  Night Owl Grinder | Late-session, task-oriented |
|  Casual Learner | Moderate, consistent study habits |
|  Distracted Drifter | High idle ratio, low efficiency |
