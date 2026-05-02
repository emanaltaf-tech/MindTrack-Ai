"""
generate_dataset.py
-------------------
Adaptive AI-Based Study Productivity System

REAL DATA MODE (recommended):
  1. Download the Kaggle dataset:
     https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset
  2. Rename the downloaded file to  kaggle_raw.csv
  3. Place it in this project folder
  4. Run:  python app.py   (auto-detects and uses real data)

FALLBACK MODE (if kaggle_raw.csv is not present):
  Generates an improved synthetic dataset using non-linear, research-backed
  relationships instead of a single explicit formula. This avoids the
  "circular formula" problem.

Column schema (both modes produce identical output):
  study_time       - minutes of active focused study
  break_time       - minutes of intentional breaks
  tasks_completed  - number of tasks/problems finished
  idle_time        - minutes of distraction / idle time
  session_time     - total session duration (minutes)
  time_of_day      - morning / afternoon / evening / night
  focus_score      - 0-100  (target for regression)
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(2024)


# ===========================================================================
#  REAL DATA MODE — Kaggle adapter
# ===========================================================================

def load_kaggle_dataset(kaggle_path: str) -> pd.DataFrame:
    """
    Load the Kaggle 'Student Habits and Academic Performance' dataset and
    map its columns to our project schema.

    Source: kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset
    Alternate: kaggle.com/datasets/lainguyn123/student-performance-factors
    """
    print(f"  [Real Data] Loading Kaggle dataset from: {kaggle_path}")
    raw = pd.read_csv(kaggle_path)
    print(f"  Raw shape : {raw.shape}")
    print(f"  Raw cols  : {list(raw.columns)}")

    # Normalise column names to lowercase + underscores
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    df = pd.DataFrame()

    # -- study_time ----------------------------------------------------------
    for col in ["study_hours_per_day", "study_hours", "hours_studied",
                "study_time_hours", "daily_study_hours"]:
        if col in raw.columns:
            df["study_time"] = (raw[col] * 60).round().astype(int).clip(5, 480)
            break
    if "study_time" not in df.columns:
        raise KeyError("Cannot find a study-hours column. "
                       "Check your CSV has a column like 'study_hours_per_day'.")

    # -- idle_time (social media + entertainment) ----------------------------
    idle_hours = pd.Series(0.0, index=raw.index)
    for col in ["social_media_hours", "social_media_usage",
                "netflix_hours", "entertainment_hours",
                "screen_time_hours", "leisure_time_hours"]:
        if col in raw.columns:
            idle_hours += raw[col].fillna(0)
    df["idle_time"] = (idle_hours * 60).round().astype(int).clip(0, 180)

    # -- break_time ----------------------------------------------------------
    for col in ["break_time", "break_minutes", "rest_time"]:
        if col in raw.columns:
            df["break_time"] = raw[col].round().astype(int).clip(0, 120)
            break
    if "break_time" not in df.columns:
        df["break_time"] = (df["study_time"] * 0.15).round().astype(int).clip(5, 90)

    # -- tasks_completed -----------------------------------------------------
    for col in ["assignments_completed", "tasks_completed",
                "assignment_completion", "num_tasks"]:
        if col in raw.columns:
            df["tasks_completed"] = raw[col].round().astype(int).clip(0, 20)
            break
    if "tasks_completed" not in df.columns:
        att_col = next((c for c in ["attendance_percentage", "attendance",
                                    "attendance_rate"] if c in raw.columns), None)
        if att_col:
            df["tasks_completed"] = ((raw[att_col] / 100) * 12).round().astype(int).clip(0, 12)
        else:
            df["tasks_completed"] = RNG.integers(2, 8, size=len(raw))

    # -- session_time --------------------------------------------------------
    df["session_time"] = (
        df["study_time"] + df["break_time"] + df["idle_time"]
        + RNG.integers(5, 20, size=len(raw))
    ).clip(10, 600)

    # -- time_of_day ---------------------------------------------------------
    for col in ["time_of_day", "study_time_of_day", "preferred_study_time"]:
        if col in raw.columns:
            mapping = {
                "morning": "morning", "afternoon": "afternoon",
                "evening": "evening", "night": "night",
                "Morning": "morning", "Afternoon": "afternoon",
                "Evening": "evening", "Night": "night",
            }
            df["time_of_day"] = raw[col].map(mapping).fillna("afternoon")
            break
    if "time_of_day" not in df.columns:
        sleep_col = next((c for c in ["sleep_hours", "sleep_duration",
                                      "hours_of_sleep"] if c in raw.columns), None)
        if sleep_col:
            def _sleep_to_tod(s):
                if s >= 8:   return "morning"
                elif s >= 7: return "afternoon"
                elif s >= 6: return "evening"
                else:        return "night"
            df["time_of_day"] = raw[sleep_col].apply(_sleep_to_tod)
        else:
            df["time_of_day"] = RNG.choice(
                ["morning", "afternoon", "evening", "night"],
                size=len(raw), p=[0.30, 0.30, 0.25, 0.15]
            )

    # -- focus_score (target) ------------------------------------------------
    for col in ["exam_score", "final_grade", "gpa", "grade",
                "academic_performance", "score", "marks"]:
        if col in raw.columns:
            vals = raw[col]
            if vals.max() <= 4.5:          # GPA scale -> convert to 0-100
                vals = (vals / 4.0) * 100
            df["focus_score"] = vals.clip(0, 100).round(1)
            break
    if "focus_score" not in df.columns:
        raise KeyError("Cannot find a score/grade target column in the dataset.")

    df = df.dropna().reset_index(drop=True)
    print(f"  Mapped shape: {df.shape}")
    print(df.describe().round(2).to_string())
    return df


# ===========================================================================
#  FALLBACK — Improved synthetic data (non-linear, non-circular)
# ===========================================================================

def _generate_synthetic(n_total: int = 1200) -> pd.DataFrame:
    """
    Improved synthetic data using NON-LINEAR, RESEARCH-BACKED relationships.

    Key improvements over the original version:
    1. Diminishing returns on study time (real cognitive science finding)
    2. Pomodoro interaction: breaks only help if study block > 25 min
    3. Individual variation via hidden person-level random effects
       (model must learn from observable features, not a memorised formula)
    4. Non-linear idle penalty: mild vs severe distraction differ
    5. Task difficulty varies across students (hidden variable)
    """

    PROFILES = {
        "deep_worker":      dict(n=350, tod=["morning", "afternoon"],
                                 study=(90, 200), break_=(10, 40), tasks=(7, 14),
                                 idle=(5, 25),   extra=(20, 60)),
        "night_owl":        dict(n=280, tod=["night", "evening"],
                                 study=(60, 150), break_=(20, 60), tasks=(4, 10),
                                 idle=(15, 55),  extra=(20, 40)),
        "afternoon_casual": dict(n=280, tod=["afternoon", "evening"],
                                 study=(30, 100), break_=(25, 80), tasks=(2, 8),
                                 idle=(20, 70),  extra=(10, 30)),
        "scattered":        dict(n=290, tod=["morning","afternoon","evening","night"],
                                 study=(15, 80),  break_=(40, 120), tasks=(0, 5),
                                 idle=(35, 110), extra=(5, 20)),
    }

    TOD_ENC = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
    records  = []

    for pname, cfg in PROFILES.items():
        n          = cfg["n"]
        tod_arr    = RNG.choice(cfg["tod"], size=n)
        study      = RNG.integers(*cfg["study"],  size=n).astype(float)
        break_     = RNG.integers(*cfg["break_"], size=n).astype(float)
        tasks      = RNG.integers(*cfg["tasks"],  size=n)
        idle       = RNG.integers(*cfg["idle"],   size=n).astype(float)
        extra      = RNG.integers(*cfg["extra"],  size=n).astype(float)
        session    = study + break_ + idle + extra
        difficulty = RNG.choice([1, 2, 3], size=n, p=[0.30, 0.50, 0.20])
        person_fx  = RNG.normal(0, 10, n)      # hidden individual variation

        for i in range(n):
            st, br, tk, il, se, tod = (
                study[i], break_[i], tasks[i], idle[i], session[i], tod_arr[i]
            )

            # 1. Diminishing returns on study time
            if st <= 90:
                study_eff = (st / 90) * 38
            else:
                study_eff = 38 + (1 - np.exp(-(st - 90) / 120)) * 10

            # 2. Break only helps if study block is meaningful
            if st > 25 and br > 0:
                optimal = st * 0.20
                break_eff = 8 * np.exp(-0.5 * ((br - optimal) / 20) ** 2)
            else:
                break_eff = -br * 0.08

            # 3. Task contribution scaled by hidden difficulty
            task_eff = (tk / difficulty[i]) * 9

            # 4. Non-linear idle penalty
            idle_ratio = il / max(se, 1)
            idle_pen   = 28 * (idle_ratio ** 0.7)

            # 5. Time-of-day (mild effect)
            tod_bonus = {0: 6, 1: 2, 2: 0, 3: -5}[TOD_ENC[tod]]

            raw_score = (8 + study_eff + break_eff + task_eff
                         - idle_pen + tod_bonus + person_fx[i])
            noise     = RNG.normal(0, 3.5)

            records.append({
                "study_time":      int(st),
                "break_time":      int(br),
                "tasks_completed": int(tk),
                "idle_time":       int(il),
                "session_time":    int(se),
                "time_of_day":     tod,
                "focus_score":     round(float(np.clip(raw_score + noise, 0, 100)), 1),
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ===========================================================================
#  Public entry point
# ===========================================================================

def build_dataset(save_path: str = "dataset.csv",
                  kaggle_path: str = None) -> pd.DataFrame:
    """
    Build the dataset. Priority:
      1. kaggle_path provided and file exists  -> real data
      2. kaggle_raw.csv in project folder      -> real data
      3. Neither found                         -> improved synthetic fallback
    """
    if kaggle_path is None:
        auto = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_raw.csv")
        if os.path.exists(auto):
            kaggle_path = auto

    if kaggle_path and os.path.exists(kaggle_path):
        try:
            df     = load_kaggle_dataset(kaggle_path)
            source = "REAL DATA (Kaggle)"
        except Exception as e:
            print(f"  [Warning] Kaggle load failed: {e}\n  Falling back to synthetic.")
            df     = _generate_synthetic()
            source = "SYNTHETIC (fallback)"
    else:
        print("  [Info] kaggle_raw.csv not found — using improved synthetic data.")
        print("         To use real data: download from Kaggle and rename to kaggle_raw.csv\n")
        df     = _generate_synthetic()
        source = "SYNTHETIC (improved non-linear)"

    df.to_csv(save_path, index=False)
    print(f"\n  Dataset source : {source}")
    print(f"  Saved          : {save_path}  ({len(df)} rows)\n")
    return df


if __name__ == "__main__":
    build_dataset()
