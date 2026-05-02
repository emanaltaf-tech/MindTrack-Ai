"""
preprocessing.py
----------------
Handles all data preparation:
  - Engineer derived features (study_efficiency, idle_ratio, break_ratio)
  - One-hot encode  time_of_day  (drop_first=True avoids the dummy variable trap)
  - Scale numerical features (StandardScaler)
  - Train / test split
  - Expose the feature column list for consistent downstream use
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Columns that go into every model
NUMERIC_FEATURES = [
    "study_time", "break_time", "tasks_completed",
    "idle_time", "session_time",
    # Derived features — computed in add_engineered_features()
    "study_efficiency",   # study_time / session_time  (how focused the session was)
    "idle_ratio",         # idle_time  / session_time  (distraction load)
    "break_ratio",        # break_time / study_time    (recovery rate)
]
CATEGORICAL_FEATURE   = "time_of_day"
TARGET_REGRESSION     = "focus_score"
TARGET_CLASSIFICATION = "productive"     # derived below


def add_labels(df: pd.DataFrame, threshold: float = 60.0) -> pd.DataFrame:
    """Add binary productivity label: 1 if focus_score >= threshold."""
    df = df.copy()
    df[TARGET_CLASSIFICATION] = (df[TARGET_REGRESSION] >= threshold).astype(int)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute three domain-relevant derived features.

    These capture *ratios* that are more informative than the raw minute
    counts alone — the model sees the same signal at different session lengths.
    """
    df = df.copy()
    df["study_efficiency"] = df["study_time"] / df["session_time"].clip(lower=1)
    df["idle_ratio"]       = df["idle_time"]  / df["session_time"].clip(lower=1)
    df["break_ratio"]      = df["break_time"] / df["study_time"].clip(lower=1)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode time_of_day.

    drop_first=True drops one dummy column (morning) to avoid perfect
    multicollinearity (the dummy variable trap).  Tree-based models are
    not affected by multicollinearity, but this is best practice and keeps
    the feature matrix full-rank for any future linear baselines.
    """
    return pd.get_dummies(df, columns=[CATEGORICAL_FEATURE], drop_first=True)


def get_feature_columns(df_encoded: pd.DataFrame) -> list[str]:
    """Return all feature column names (numeric + encoded categorical)."""
    tod_cols = [c for c in df_encoded.columns if c.startswith("time_of_day_")]
    return NUMERIC_FEATURES + tod_cols


def preprocess(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train_sc, X_test_sc  : scaled feature arrays (numpy)
    y_reg_train, y_reg_test: regression targets
    y_cls_train, y_cls_test: classification targets
    scaler                 : fitted StandardScaler (for inference)
    feature_cols           : list of feature column names
    df_encoded             : encoded DataFrame (useful for clustering)
    """
    df = add_labels(df)
    df = add_engineered_features(df)
    df = encode_categorical(df)

    feature_cols = get_feature_columns(df)

    X     = df[feature_cols].values
    y_reg = df[TARGET_REGRESSION].values
    y_cls = df[TARGET_CLASSIFICATION].values

    # Split — stratified on classification target keeps class balance consistent
    X_train, X_test, \
    y_reg_train, y_reg_test, \
    y_cls_train, y_cls_test = train_test_split(
        X, y_reg, y_cls, test_size=test_size, random_state=random_state,
        stratify=y_cls,
    )

    # Scale — fit ONLY on training data to prevent data leakage
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return (
        X_train_sc, X_test_sc,
        y_reg_train, y_reg_test,
        y_cls_train, y_cls_test,
        scaler, feature_cols, df
    )


def encode_single_input(
    study_time: int, break_time: int, tasks_completed: int,
    idle_time: int, session_time: int, time_of_day: str,
    scaler: StandardScaler, feature_cols: list[str]
) -> np.ndarray:
    """
    Encode and scale a single user input row for inference.
    Mirrors the training encoding exactly (including derived features).
    """
    # Derived features — must match add_engineered_features()
    study_efficiency = study_time / max(session_time, 1)
    idle_ratio       = idle_time  / max(session_time, 1)
    break_ratio      = break_time / max(study_time,   1)

    # drop_first=True in training drops 'morning', so only afternoon/evening/night dummies exist
    tod_options_kept = ["afternoon", "evening", "night"]

    row = {
        "study_time":       study_time,
        "break_time":       break_time,
        "tasks_completed":  tasks_completed,
        "idle_time":        idle_time,
        "session_time":     session_time,
        "study_efficiency": study_efficiency,
        "idle_ratio":       idle_ratio,
        "break_ratio":      break_ratio,
    }
    for opt in tod_options_kept:
        row[f"time_of_day_{opt}"] = 1 if time_of_day == opt else 0

    # Align to training feature order — unknown columns default to 0
    vec = np.array([row.get(c, 0) for c in feature_cols], dtype=float).reshape(1, -1)
    return scaler.transform(vec)
