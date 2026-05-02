"""
charts.py
---------
All matplotlib/plotly chart generation for the Gradio UI.
Every function returns a matplotlib Figure (safe for Gradio gr.Plot).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

# ── Palette ──────────────────────────────────────────────────────────────────
DARK_BG   = "#0F1117"
CARD_BG   = "#1A1D27"
ACCENT1   = "#6C63FF"
ACCENT2   = "#00D4AA"
ACCENT3   = "#FF6B6B"
ACCENT4   = "#FFD166"
TEXT_MAIN = "#E8EAF0"
TEXT_SUB  = "#8B8FA8"
GRID_CLR  = "#252836"

CLUSTER_COLORS = ["#4F86C6", "#7B5EA7", "#5BAD92", "#E07B54"]

def _apply_dark_style(fig, axes=None):
    fig.patch.set_facecolor(DARK_BG)
    if axes is None:
        axes = fig.get_axes()
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_SUB, labelsize=9)
        ax.xaxis.label.set_color(TEXT_SUB)
        ax.yaxis.label.set_color(TEXT_SUB)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(color=GRID_CLR, linewidth=0.6, linestyle="--", alpha=0.7)


# ── 1. Score Gauge ────────────────────────────────────────────────────────────

def score_gauge(focus_score: int, productive: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    # Background arc
    theta_bg = np.linspace(np.pi, 0, 300)
    ax.plot(theta_bg, [1]*300, color=GRID_CLR, linewidth=18, solid_capstyle="round")

    # Score arc
    frac = focus_score / 100
    color = ACCENT2 if productive else ACCENT3
    if focus_score < 40:
        color = ACCENT3
    elif focus_score < 65:
        color = ACCENT4
    else:
        color = ACCENT2

    theta_score = np.linspace(np.pi, np.pi - frac * np.pi, 300)
    ax.plot(theta_score, [1]*300, color=color, linewidth=18,
            solid_capstyle="round",
            path_effects=[pe.Stroke(linewidth=22, foreground=DARK_BG), pe.Normal()])

    # Score text
    ax.text(0, 0.08, str(focus_score), ha="center", va="center",
            fontsize=42, fontweight="bold", color=TEXT_MAIN,
            fontfamily="DejaVu Sans")
    ax.text(0, -0.28, "FOCUS SCORE", ha="center", va="center",
            fontsize=9, color=TEXT_SUB, fontfamily="DejaVu Sans")

    label = "✓ PRODUCTIVE" if productive else "✗ NON-PRODUCTIVE"
    lcolor = ACCENT2 if productive else ACCENT3
    ax.text(0, -0.52, label, ha="center", va="center",
            fontsize=10, fontweight="bold", color=lcolor)

    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, np.pi)
    ax.axis("off")
    plt.tight_layout(pad=0.5)
    return fig


# ── 2. Score Breakdown Bar ────────────────────────────────────────────────────

def score_breakdown(study_time, break_time, tasks_completed, idle_time,
                    session_time, focus_score) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    _apply_dark_style(fig, ax)

    study_norm = min(study_time / 180, 1.0)
    task_norm  = min(tasks_completed / 12, 1.0)
    idle_pen   = min(idle_time / max(session_time, 1), 0.6)
    brk_ratio  = min(break_time / max(study_time, 1), 0.5)

    labels = ["Study\nContrib.", "Task\nContrib.", "Idle\nPenalty", "Break\nPenalty"]
    values = [study_norm*42, task_norm*35, -(idle_pen*25), -(brk_ratio*12)]
    colors = [ACCENT2, ACCENT1, ACCENT3, ACCENT4]

    bars = ax.bar(labels, values, color=colors, width=0.55,
                  edgecolor=DARK_BG, linewidth=1.5)

    for bar, val in zip(bars, values):
        ypos = val + (1.5 if val >= 0 else -2.5)
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=11, fontweight="bold", color=TEXT_MAIN)

    ax.axhline(0, color=TEXT_SUB, linewidth=0.8)
    ax.set_title(f"Score Component Breakdown  ·  Final: {focus_score}/100",
                 color=TEXT_MAIN, fontsize=12, pad=10, fontweight="bold")
    ax.set_ylabel("Points", color=TEXT_SUB)
    ax.set_ylim(min(values) - 10, max(values) + 12)
    plt.tight_layout()
    return fig


# ── 3. Session Gantt Chart ────────────────────────────────────────────────────

def session_gantt(study_time, break_time, idle_time, session_time) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.6))
    _apply_dark_style(fig, ax)
    ax.grid(False)

    segments = [
        ("Study",       study_time,  ACCENT2),
        ("Break",       break_time,  ACCENT4),
        ("Idle / Lost", idle_time,   ACCENT3),
    ]
    # remainder
    accounted = study_time + break_time + idle_time
    other = max(0, session_time - accounted)
    if other > 0:
        segments.append(("Other", other, TEXT_SUB))

    x = 0
    for label, dur, color in segments:
        if dur <= 0:
            continue
        bar = FancyBboxPatch((x, 0.3), dur, 0.4,
                             boxstyle="round,pad=2",
                             facecolor=color, edgecolor=DARK_BG, linewidth=2)
        ax.add_patch(bar)
        if dur > session_time * 0.07:
            ax.text(x + dur/2, 0.5, f"{label}\n{dur}m",
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color=DARK_BG)
        x += dur

    ax.set_xlim(0, max(session_time, x) * 1.02)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Minutes", color=TEXT_SUB)
    ax.set_yticks([])
    ax.set_title("Session Timeline (Gantt View)", color=TEXT_MAIN,
                 fontsize=12, fontweight="bold", pad=8)

    # Add time markers
    for marker in range(0, int(x)+1, 30):
        ax.axvline(marker, color=GRID_CLR, linewidth=0.8, linestyle=":")

    plt.tight_layout()
    return fig


# ── 4. Feature Importance ────────────────────────────────────────────────────

def feature_importance_chart(fi_df: pd.DataFrame) -> plt.Figure:
    top = fi_df.head(8).copy()
    top["feature"] = top["feature"].str.replace("time_of_day_", "ToD: ").str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    _apply_dark_style(fig, ax)

    colors = [ACCENT1 if i < 3 else ACCENT2 if i < 6 else TEXT_SUB
              for i in range(len(top))]
    bars = ax.barh(top["feature"][::-1], top["importance"][::-1],
                   color=colors[::-1], height=0.6,
                   edgecolor=DARK_BG, linewidth=1)

    for bar, val in zip(bars, top["importance"][::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9, color=TEXT_MAIN)

    ax.set_xlabel("Importance", color=TEXT_SUB)
    ax.set_title("Feature Importance (Random Forest)", color=TEXT_MAIN,
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xlim(0, top["importance"].max() * 1.25)
    plt.tight_layout()
    return fig


# ── 5. Confusion Matrix ──────────────────────────────────────────────────────

def confusion_matrix_chart(cm: list) -> plt.Figure:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    _apply_dark_style(fig, ax)

    im = ax.imshow(cm_arr, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=cm_arr.max())
    labels = ["Non-Prod.", "Productive"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, color=TEXT_SUB)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, color=TEXT_SUB)
    ax.set_xlabel("Predicted", color=TEXT_SUB)
    ax.set_ylabel("Actual", color=TEXT_SUB)
    ax.set_title("Confusion Matrix", color=TEXT_MAIN, fontsize=12,
                 fontweight="bold", pad=8)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    fontsize=20, fontweight="bold",
                    color=DARK_BG if cm_arr[i, j] > cm_arr.max()/2 else TEXT_MAIN)

    plt.tight_layout()
    return fig


# ── 6. Per-fold Cross-Validation Chart ───────────────────────────────────────

def cv_folds_chart(cv_scores: dict) -> plt.Figure:
    """
    Visualise per-fold R² and Accuracy scores from 5-fold cross-validation.
    Shows each fold individually so the teacher can see variance, not just mean.
    """
    reg_folds = cv_scores.get("reg_folds", [])
    cls_folds = cv_scores.get("cls_folds", [])

    if not reg_folds or not cls_folds:
        fig, ax = plt.subplots(figsize=(7, 3))
        _apply_dark_style(fig, ax)
        ax.text(0.5, 0.5, "CV data not available", ha="center", va="center",
                color=TEXT_MAIN, transform=ax.transAxes)
        return fig

    folds = [f"Fold {i+1}" for i in range(len(reg_folds))]
    x = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 3.8))
    _apply_dark_style(fig, ax)

    bars1 = ax.bar(x - width/2, reg_folds, width, label="R² (Regressor)",
                   color=ACCENT2, edgecolor=DARK_BG, linewidth=1.5)
    bars2 = ax.bar(x + width/2, cls_folds, width, label="Accuracy (Classifier)",
                   color=ACCENT1, edgecolor=DARK_BG, linewidth=1.5)

    for bar, val in zip(bars1, reg_folds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=8, color=TEXT_MAIN)
    for bar, val in zip(bars2, cls_folds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=8, color=TEXT_MAIN)

    reg_mean = np.mean(reg_folds)
    cls_mean = np.mean(cls_folds)
    ax.axhline(reg_mean, color=ACCENT2, linewidth=1.2, linestyle="--", alpha=0.7,
               label=f"R² mean = {reg_mean:.3f}")
    ax.axhline(cls_mean, color=ACCENT1, linewidth=1.2, linestyle="--", alpha=0.7,
               label=f"Acc mean = {cls_mean:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylabel("Score", color=TEXT_SUB)
    ax.set_ylim(0, 1.12)
    ax.set_title("5-Fold Cross-Validation — Per-Fold Scores",
                 color=TEXT_MAIN, fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_CLR,
              labelcolor=TEXT_MAIN, loc="lower right")
    plt.tight_layout()
    return fig


# ── 7. Silhouette Score vs k (Cluster Selection) ──────────────────────────────

def silhouette_k_chart(elbow_data: dict) -> plt.Figure:
    """
    Plot silhouette score for each tested k value.
    This shows why the chosen k was selected — the best silhouette wins.
    """
    if not elbow_data:
        fig, ax = plt.subplots(figsize=(5, 3))
        _apply_dark_style(fig, ax)
        return fig

    ks   = sorted(elbow_data.keys())
    sils = [elbow_data[k] for k in ks]
    best_k = max(elbow_data, key=elbow_data.get)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _apply_dark_style(fig, ax)

    colors = [ACCENT2 if k == best_k else ACCENT1 for k in ks]
    bars = ax.bar([str(k) for k in ks], sils, color=colors, width=0.5,
                  edgecolor=DARK_BG, linewidth=1.5)

    for bar, sil, k in zip(bars, sils, ks):
        label = f"{sil:.4f}" + (" ← selected" if k == best_k else "")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                label, ha="center", va="bottom", fontsize=9,
                color=ACCENT2 if k == best_k else TEXT_MAIN, fontweight="bold")

    ax.set_xlabel("Number of Clusters (k)", color=TEXT_SUB)
    ax.set_ylabel("Silhouette Score", color=TEXT_SUB)
    ax.set_title("Cluster Selection: Silhouette Score per k",
                 color=TEXT_MAIN, fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim(0, max(sils) * 1.25)
    plt.tight_layout()
    return fig


# ── 8. Cluster distribution ───────────────────────────────────────────────────

def cluster_distribution(labels: np.ndarray, cluster_map: dict) -> plt.Figure:
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    _apply_dark_style(fig, ax)

    names  = [cluster_map.get(u, ("Unknown","?","#aaa"))[0] for u in unique]
    colors = [cluster_map.get(u, ("Unknown","?","#aaa"))[2] for u in unique]
    pcts   = counts / counts.sum() * 100

    bars = ax.bar(names, pcts, color=colors, width=0.55,
                  edgecolor=DARK_BG, linewidth=2)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{pct:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TEXT_MAIN)

    ax.set_ylabel("% of Students", color=TEXT_SUB)
    ax.set_title("Learner Type Distribution", color=TEXT_MAIN,
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    plt.tight_layout()
    return fig