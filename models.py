"""
models.py
---------
Three ML models:
  1. Regression      – RandomForestRegressor  -> predict focus_score
  2. Classification  – RandomForestClassifier -> productive / non-productive
  3. Clustering      – KMeans (k=3-5)         -> behaviour pattern groups

Notes:
  - R² uses sklearn's r2_score (standard formula: 1 - SS_res/SS_tot).
  - 5-fold cross-validation scores stored for the accuracy report tab.
  - Cluster k is selected by silhouette score; elbow data also stored for
    the "Study Patterns" visualisation.
  - Cluster quality ranking uses named feature lookup instead of hard-coded
    column indices, so it remains correct if feature order changes.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score


class StudyProductivityModel:

    def __init__(self):
        self.regressor   = None
        self.classifier  = None
        self.kmeans      = None
        self.cluster_map = {}
        self.feature_cols: list[str] = []   # set during clustering

        self.reg_metrics        = {}
        self.cls_metrics        = {}
        self.cluster_silhouette = None
        self.elbow_data         = {}   # {k: silhouette_score} for all tested k values
        self.cv_scores          = {}   # cross-validation results

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, X_train_sc, X_test_sc,
              y_reg_train, y_reg_test,
              y_cls_train, y_cls_test,
              X_all_sc,
              feature_cols: list[str] | None = None):
        if feature_cols:
            self.feature_cols = feature_cols
        self._train_regression(X_train_sc, X_test_sc, y_reg_train, y_reg_test)
        self._train_classification(X_train_sc, X_test_sc, y_cls_train, y_cls_test)
        self._train_clustering(X_all_sc)
        self._cross_validate(X_train_sc, y_reg_train, y_cls_train)

    def _train_regression(self, X_tr, X_te, y_tr, y_te):
        self.regressor = RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_leaf=3, random_state=42, n_jobs=-1,
        )
        self.regressor.fit(X_tr, y_tr)
        pred = self.regressor.predict(X_te)
        self.reg_metrics = {
            "MAE":  round(mean_absolute_error(y_te, pred), 2),
            "RMSE": round(float(mean_squared_error(y_te, pred) ** 0.5), 2),
            "R2":   round(float(r2_score(y_te, pred)), 4),
        }
        print(f"  Regression  ->  MAE={self.reg_metrics['MAE']}  "
              f"RMSE={self.reg_metrics['RMSE']}  R2={self.reg_metrics['R2']}")

    def _train_classification(self, X_tr, X_te, y_tr, y_te):
        self.classifier = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=3, random_state=42, n_jobs=-1,
            class_weight="balanced",
        )
        self.classifier.fit(X_tr, y_tr)
        pred = self.classifier.predict(X_te)
        self.cls_metrics = {
            "accuracy":  round(accuracy_score(y_te, pred), 4),
            "report":    classification_report(y_te, pred,
                             target_names=["Non-Productive", "Productive"]),
            "confusion": confusion_matrix(y_te, pred).tolist(),
        }
        print(f"  Classification  ->  Accuracy={self.cls_metrics['accuracy']:.2%}")

    def _train_clustering(self, X_all_sc):
        """
        Try k = 3, 4, 5 clusters. Select the k with the highest silhouette score.
        All silhouette values are stored in self.elbow_data for the elbow chart.
        """
        best_k, best_sil, best_km = 4, -1, None
        for k in range(3, 6):
            km     = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=15)
            labels = km.fit_predict(X_all_sc)
            sil    = float(silhouette_score(X_all_sc, labels, sample_size=min(500, len(labels))))
            self.elbow_data[k] = round(sil, 4)
            if sil > best_sil:
                best_k, best_sil, best_km = k, sil, km

        self.kmeans             = best_km
        self.cluster_silhouette = round(best_sil, 4)
        self._assign_cluster_names(X_all_sc)
        print(f"  Clustering (k={best_k})  ->  Silhouette={best_sil:.3f}")
        print(f"  Elbow data: {self.elbow_data}")

    def _cross_validate(self, X_tr, y_reg_tr, y_cls_tr):
        """5-fold CV on the training set — honest estimate free of test-set leakage."""
        reg_cv = cross_val_score(
            RandomForestRegressor(n_estimators=100, max_depth=10,
                                  min_samples_leaf=3, random_state=42, n_jobs=-1),
            X_tr, y_reg_tr, cv=5, scoring="r2"
        )
        cls_cv = cross_val_score(
            RandomForestClassifier(n_estimators=100, max_depth=10,
                                   min_samples_leaf=3, random_state=42, n_jobs=-1,
                                   class_weight="balanced"),
            X_tr, y_cls_tr, cv=5, scoring="accuracy"
        )
        self.cv_scores = {
            "reg_r2_mean":  round(float(reg_cv.mean()), 4),
            "reg_r2_std":   round(float(reg_cv.std()),  4),
            "cls_acc_mean": round(float(cls_cv.mean()), 4),
            "cls_acc_std":  round(float(cls_cv.std()),  4),
            "reg_folds":    [round(float(v), 4) for v in reg_cv],
            "cls_folds":    [round(float(v), 4) for v in cls_cv],
        }
        print(f"  5-fold CV  ->  R2={self.cv_scores['reg_r2_mean']} "
              f"(+/-{self.cv_scores['reg_r2_std']})  "
              f"Acc={self.cv_scores['cls_acc_mean']} "
              f"(+/-{self.cv_scores['cls_acc_std']})")

    def _assign_cluster_names(self, X_all_sc):
        """
        Rank clusters by quality using named feature lookup.

        'Quality' = high study efficiency + many tasks − high idle ratio.
        We look up column indices by name from self.feature_cols so the
        formula stays correct if the feature order ever changes.
        """
        centers = self.kmeans.cluster_centers_
        n       = len(centers)

        if self.feature_cols:
            idx_study_eff = (self.feature_cols.index("study_efficiency")
                             if "study_efficiency" in self.feature_cols else 0)
            idx_idle      = (self.feature_cols.index("idle_ratio")
                             if "idle_ratio"       in self.feature_cols else 3)
            idx_tasks     = (self.feature_cols.index("tasks_completed")
                             if "tasks_completed"  in self.feature_cols else 2)
            quality = (centers[:, idx_study_eff]
                       - centers[:, idx_idle]
                       + centers[:, idx_tasks] * 0.1)
        else:
            # Fallback: raw indices (study_time, break_time, tasks, idle, session)
            quality = centers[:, 0] - centers[:, 3] + centers[:, 2]

        sorted_q = np.argsort(quality)[::-1]

        name_pool = [
            ("Deep Focus Worker",  "🧠", "#4F86C6"),
            ("Night Owl Grinder",  "🌙", "#7B5EA7"),
            ("Casual Learner",     "📚", "#5BAD92"),
            ("Distracted Drifter", "⚡", "#E07B54"),
        ]

        assigned = {}
        assigned[sorted_q[0]]  = name_pool[0]
        assigned[sorted_q[-1]] = name_pool[3]
        remaining = [name_pool[1], name_pool[2]]
        ri = 0
        for idx in range(n):
            if idx not in assigned:
                assigned[idx] = remaining[ri % len(remaining)]
                ri += 1
        self.cluster_map = assigned

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, X_scaled: np.ndarray):
        focus_score  = float(self.regressor.predict(X_scaled)[0])
        focus_score  = max(0, min(100, round(focus_score)))
        productive   = bool(self.classifier.predict(X_scaled)[0])
        cluster_id   = int(self.kmeans.predict(X_scaled)[0])
        cluster_info = self.cluster_map.get(cluster_id, ("Unknown", "❓", "#aaa"))
        return focus_score, productive, cluster_id, cluster_info

    # ── Feature importance ───────────────────────────────────────────────────

    def feature_importance(self, feature_cols: list) -> pd.DataFrame:
        imp = self.regressor.feature_importances_
        df  = pd.DataFrame({"feature": feature_cols, "importance": imp})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
