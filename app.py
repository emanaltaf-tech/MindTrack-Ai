"""
app.py — MindTrack AI
Streamlined 4-screen flow:
  Landing  → Login / Register → Input Section → Results (with embedded history)
ML logic is unchanged; only the UI/flow is modified.
"""

import sys, os, csv, json, datetime, traceback, hashlib, secrets, hashlib, secrets
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_dataset import build_dataset
from preprocessing   import preprocess, encode_single_input
from models          import StudyProductivityModel
from recommendations import generate_recommendations, best_study_time
from charts          import score_gauge, session_gantt, score_breakdown


# ─────────────────────────────────────────────────────────────────────────────
# File Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
DATA_PATH    = os.path.join(BASE_DIR, "dataset.csv")
KAGGLE_PATH  = os.path.join(BASE_DIR, "kaggle_raw.csv")
USERS_PATH   = os.path.join(BASE_DIR, "users.json")
HISTORY_PATH = os.path.join(BASE_DIR, "session_history.csv")

HISTORY_COLS = [
    "date", "time", "username", "full_name", "focus_score",
    "study_time", "break_time", "tasks_completed",
    "idle_time", "session_time", "time_of_day",
    "productive", "learning_style",
]


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap & Model Training  
# ─────────────────────────────────────────────────────────────────────────────
def _bootstrap_model():
    try:
        if not os.path.exists(DATA_PATH):
            build_dataset(
                DATA_PATH,
                kaggle_path=KAGGLE_PATH if os.path.exists(KAGGLE_PATH) else None,
            )
        else:
            if os.path.exists(KAGGLE_PATH):
                if os.path.getmtime(KAGGLE_PATH) > os.path.getmtime(DATA_PATH):
                    build_dataset(DATA_PATH, kaggle_path=KAGGLE_PATH)
        df_raw = pd.read_csv(DATA_PATH)
        result = preprocess(df_raw)
        (
            X_tr, X_te,
            y_reg_tr, y_reg_te,
            y_cls_tr, y_cls_te,
            scaler, feature_cols, df_enc,
        ) = result
        X_all = scaler.transform(df_enc[feature_cols].values)
        m = StudyProductivityModel()
        m.train(
            X_tr, X_te,
            y_reg_tr, y_reg_te,
            y_cls_tr, y_cls_te,
            X_all,
            feature_cols=feature_cols,
        )
        return m, scaler, feature_cols
    except Exception as exc:
        print(f"[BOOT ERROR] {exc}")
        traceback.print_exc()
        raise

MODEL, SCALER, FEATURE_COLS = _bootstrap_model()


# ─────────────────────────────────────────────────────────────────────────────
# User Account Storage  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _load_users() -> dict:
    if not os.path.exists(USERS_PATH):
        return {}
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    try:
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[WARN] Could not save users: {exc}")


def username_exists(username: str) -> bool:
    return username.strip().lower() in _load_users()


def get_user_profile(username: str) -> dict:
    return _load_users().get(username.strip().lower(), {})


def _hash_password(password: str, salt: str = "") -> str:
    """SHA-256 hash with per-user salt for secure password storage."""
    if not salt:
        salt = secrets.token_hex(16)
    digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}:{digest}"


def _verify_password(stored: str, candidate: str) -> bool:
    """Verify a candidate password against the stored salt:hash string."""
    try:
        salt, _ = stored.split(":", 1)
        return stored == _hash_password(candidate, salt)
    except Exception:
        return False


def register_user(username, full_name, age, gender, education, field_of_study,
                  password: str) -> None:
    users = _load_users()
    users[username.strip().lower()] = {
        "username":       username.strip().lower(),
        "full_name":      full_name.strip(),
        "age":            int(age) if age else 0,
        "gender":         gender,
        "education":      education,
        "field_of_study": field_of_study.strip(),
        "password_hash":  _hash_password(password),
        "registered_at":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    _save_users(users)


# ─────────────────────────────────────────────────────────────────────────────
# Session History Storage  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def _load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=HISTORY_COLS)
    try:
        df = pd.read_csv(HISTORY_PATH)
        if "username" not in df.columns and "name" in df.columns:
            df["username"] = df["name"]
        return df
    except Exception:
        return pd.DataFrame(columns=HISTORY_COLS)


def get_user_history(username: str) -> pd.DataFrame:
    df = _load_history()
    if df.empty or not username:
        return pd.DataFrame(columns=HISTORY_COLS)
    mask = df["username"].astype(str).str.strip().str.lower() == username.strip().lower()
    return df[mask].reset_index(drop=True)


def _save_session(username, full_name, focus_score,
                  study_time, break_time, tasks_completed,
                  idle_time, session_time, time_of_day,
                  productive, cluster_name) -> None:
    now = datetime.datetime.now()
    row = [
        now.strftime("%Y-%m-%d"), now.strftime("%H:%M"),
        username.strip().lower(), full_name,
        focus_score,
        int(study_time), int(break_time), int(tasks_completed),
        int(idle_time), int(session_time), time_of_day,
        "Yes" if productive else "No", cluster_name,
    ]
    write_header = not os.path.exists(HISTORY_PATH)
    try:
        with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(HISTORY_COLS)
            writer.writerow(row)
    except Exception as exc:
        print(f"[WARN] Could not save session: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers  
# ─────────────────────────────────────────────────────────────────────────────
def _compute_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    scores   = df["focus_score"].astype(float).tolist()
    n        = len(scores)
    tod_avg  = (
        df.assign(s=df["focus_score"].astype(float))
          .groupby("time_of_day")["s"].mean()
    ) if "time_of_day" in df.columns else pd.Series(dtype=float)
    prod_pct = (
        round((df["productive"] == "Yes").sum() / n * 100)
        if "productive" in df.columns else None
    )
    return {
        "count":    n,
        "best":     int(max(scores)),
        "avg":      round(sum(scores) / n, 1),
        "last":     int(scores[-1]),
        "trend":    round(scores[-1] - scores[-2], 1) if n >= 2 else None,
        "best_tod": tod_avg.idxmax() if not tod_avg.empty else None,
        "prod_pct": prod_pct,
        "last_date": df["date"].iloc[-1] if "date" in df.columns else "—",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Alert helpers
# ─────────────────────────────────────────────────────────────────────────────
def _err(msg):  return f"<div class='alert alert-error'><span class='alert-icon'>!</span>{msg}</div>"
def _warn(msg): return f"<div class='alert alert-warn'><span class='alert-icon'>!</span>{msg}</div>"
def _ok(msg):   return f"<div class='alert alert-ok'><span class='alert-icon'>&#10003;</span>{msg}</div>"
def _info(msg): return f"<div class='alert alert-info'><span class='alert-icon'>i</span>{msg}</div>"


# ─────────────────────────────────────────────────────────────────────────────
# History chart helpers  
# ─────────────────────────────────────────────────────────────────────────────
C_DARK  = "#0F1117"; C_CARD = "#1A1D27"; C_ACC = "#2563eb"
C_GRN   = "#16a34a"; C_ORG  = "#d97706"; C_RED = "#dc2626"
C_TXT   = "#E8EAF0"; C_SUB  = "#8B8FA8"; C_GRID = "#252836"


def _dark_ax(fig, ax):
    fig.patch.set_facecolor(C_DARK)
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_SUB, labelsize=9)
    ax.xaxis.label.set_color(C_SUB)
    ax.yaxis.label.set_color(C_SUB)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_GRID)
    ax.grid(color=C_GRID, linewidth=0.6, linestyle="--", alpha=0.7)


def _trend_chart(df: pd.DataFrame, display_name: str):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    _dark_ax(fig, ax)
    if df.empty:
        ax.text(0.5, 0.5, f"No sessions yet for {display_name}.\nAnalyse a session to start tracking.",
                ha="center", va="center", color=C_SUB, fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout()
        return fig
    scores = df["focus_score"].astype(float).tolist()
    labels = [f"#{i+1}" for i in range(len(scores))]
    ax.plot(labels, scores, color=C_ACC, linewidth=2.2, zorder=2)
    ax.fill_between(labels, scores, alpha=0.08, color=C_ACC)
    for lbl, s in zip(labels, scores):
        col = C_GRN if s >= 75 else C_ORG if s >= 50 else C_RED
        ax.plot(lbl, s, "o", markersize=8, markerfacecolor=col,
                markeredgecolor="#fff", markeredgewidth=1.5, zorder=4)
    avg = sum(scores) / len(scores)
    ax.axhline(avg, color=C_SUB, linewidth=1, linestyle="--", alpha=0.6)
    ax.text(len(labels) - 0.5, avg + 2, f"Avg {avg:.0f}",
            color=C_SUB, fontsize=9, ha="right", va="bottom")
    best_i = scores.index(max(scores))
    ax.annotate(f"Best: {int(max(scores))}",
                xy=(labels[best_i], max(scores)),
                xytext=(0, 16), textcoords="offset points",
                ha="center", fontsize=9, color=C_GRN, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_GRN, lw=1))
    ax.set_ylim(0, 110)
    ax.set_xlabel("Session", color=C_SUB, fontsize=10)
    ax.set_ylabel("Focus Score", color=C_SUB, fontsize=10)
    ax.set_title(f"Focus Score History — {display_name}",
                 color=C_TXT, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def _bar_chart(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    _dark_ax(fig, ax)
    if df.empty or len(df) < 2:
        ax.text(0.5, 0.5, "Log at least 2 sessions for this chart.",
                ha="center", va="center", color=C_SUB, fontsize=12, transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout()
        return fig
    scores = df["focus_score"].astype(float).tolist()
    labels = [f"#{i+1}" for i in range(len(scores))]
    colors = [C_GRN if s >= 75 else C_ORG if s >= 50 else C_RED for s in scores]
    bars   = ax.bar(labels, scores, color=colors, width=0.55,
                    edgecolor=C_DARK, linewidth=1.2)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                str(int(val)), ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C_TXT)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Session", color=C_SUB, fontsize=10)
    ax.set_ylabel("Focus Score", color=C_SUB, fontsize=10)
    ax.set_title("Session-by-Session Breakdown",
                 color=C_TXT, fontsize=13, fontweight="bold", pad=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_GRN, label="Strong Focus (75+)"),
        Patch(facecolor=C_ORG, label="Moderate (50-74)"),
        Patch(facecolor=C_RED, label="Needs Attention (<50)"),
    ], facecolor=C_CARD, edgecolor=C_GRID, labelcolor=C_SUB, fontsize=8, loc="upper left")
    plt.tight_layout()
    return fig


def _history_table_html(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    rows = ""
    for _, r in df.iloc[::-1].head(30).iterrows():
        s   = int(float(r.get("focus_score", 0)))
        sc  = C_GRN if s >= 75 else C_ORG if s >= 50 else C_RED
        prd = r.get("productive", "")
        pc  = "#16a34a" if prd == "Yes" else "#dc2626"
        rows += f"""<tr>
          <td class="ht">{r.get('date','')}</td>
          <td class="ht">{r.get('time','')}</td>
          <td class="ht" style="color:{sc};font-weight:700;">{s}</td>
          <td class="ht">{r.get('study_time','')} min</td>
          <td class="ht">{r.get('tasks_completed','')}</td>
          <td class="ht">{str(r.get('time_of_day','')).capitalize()}</td>
          <td class="ht" style="color:{pc};font-weight:600;">{prd}</td>
          <td class="ht">{r.get('learning_style','')}</td>
        </tr>"""
    return f"""
    <div style="margin-top:32px;">
      <h3 style="font-size:15px;font-weight:700;color:#0f172a;
          border-bottom:1px solid #e2e8f0;padding-bottom:10px;margin-bottom:16px;">
        📋 Complete Session Log
      </h3>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;font-size:13px;font-family:'Inter',sans-serif;">
          <thead><tr style="background:#f1f5f9;">
            <th class="hh">Date</th><th class="hh">Time</th>
            <th class="hh">Focus Score</th><th class="hh">Study Time</th>
            <th class="hh">Tasks</th><th class="hh">Time of Day</th>
            <th class="hh">Productive</th><th class="hh">Learning Style</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>
    <style>
      .hh {{padding:10px 12px;text-align:left;border:1px solid #e2e8f0;
            color:#475569;font-weight:600;font-size:12px;white-space:nowrap;}}
      .ht {{padding:10px 12px;border:1px solid #e2e8f0;color:#0f172a;white-space:nowrap;}}
    </style>"""


# ─────────────────────────────────────────────────────────────────────────────
# AUTH HANDLERS  
# ─────────────────────────────────────────────────────────────────────────────

def handle_login(username: str, password: str):
    """Validate username + password → go to Input page on success."""
    u = (username or "").strip().lower()
    p = (password or "").strip()

    if not u:
        return (
            gr.update(value=_err("Please enter your username or email."), visible=True),
            gr.Tabs(selected=1),
            "",
        )
    if not p:
        return (
            gr.update(value=_err("Please enter your password."), visible=True),
            gr.Tabs(selected=1),
            "",
        )
    if not username_exists(u):
        return (
            gr.update(
                value=_err(
                    "No account found for <b>" + u + "</b>. "
                    "Please check your username or "
                    "<b>register a new account</b> below."
                ),
                visible=True,
            ),
            gr.Tabs(selected=1),
            "",
        )
    profile = get_user_profile(u)
    stored  = profile.get("password_hash", "")
    # Legacy accounts (no password set): block login, prompt re-register
    if not stored:
        return (
            gr.update(
                value=_err("This account was created before password auth was added. "
                           "Please register a new account."),
                visible=True,
            ),
            gr.Tabs(selected=1),
            "",
        )
    if not _verify_password(stored, p):
        return (
            gr.update(
                value=_err(
                    "Incorrect password. Please try again. "
                    "<br><small style='color:#9b1c1c;'>Tip: Check caps lock and try again, "
                    "or register a new account if you've forgotten your password.</small>"
                ),
                visible=True,
            ),
            gr.Tabs(selected=1),
            "",
        )
    return (
        gr.update(value="", visible=False),
        gr.Tabs(selected=2),
        u,
    )


def handle_register(username, full_name, age, gender, education, field_of_study,
                    password, confirm_password):
    """Validate & create account (with password), then redirect to Login."""
    u  = (username or "").strip().lower()
    pw = (password or "").strip()
    cp = (confirm_password or "").strip()

    errors = []
    if not u:
        errors.append("Username is required.")
    elif len(u) < 3:
        errors.append("Username must be at least 3 characters.")
    elif not u.replace("_","").replace("-","").replace(".","").replace("@","").isalnum():
        errors.append("Username may only contain letters, numbers, and . _ - @ characters.")
    elif username_exists(u):
        errors.append(f"Username <b>{u}</b> is already taken. Please choose another.")
    if not full_name or not full_name.strip():
        errors.append("Full name is required.")
    if age is None or age == "":
        errors.append("Age is required.")
    elif float(age) < 13 or float(age) > 100:
        errors.append("Age must be between 13 and 100.")
    if not gender:
        errors.append("Please select a gender.")
    if not education:
        errors.append("Please select your education level.")
    if not pw:
        errors.append("Password is required.")
    elif len(pw) < 6:
        errors.append("Password must be at least 6 characters.")
    elif pw != cp:
        errors.append("Passwords do not match.")

    if errors:
        msg = "<br>".join(f"&bull; {e}" for e in errors)
        return (
            gr.update(value=_err(f"Please fix the following:<br>{msg}"), visible=True),
            gr.Tabs(selected=5),
        )

    try:
        register_user(u, full_name, age, gender, education, field_of_study or "", pw)
    except Exception as exc:
        return (
            gr.update(value=_err(f"Could not save account. ({exc})"), visible=True),
            gr.Tabs(selected=5),
        )

    return (
        gr.update(
            value=_ok(f"Account created for <b>{full_name.strip()}</b>! "
                      "Please log in with your username and password."),
            visible=True,
        ),
        gr.Tabs(selected=1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS  
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis(
    current_user,
    study_time, break_time, tasks_completed,
    idle_time, session_time, time_of_day,
    study_goal, pref_time, sleep_hours, stress_level,
):
    if not current_user or not current_user.strip():
        raise gr.Error("You must log in before analysing a session.")

    profile   = get_user_profile(current_user.strip())
    full_name = profile.get("full_name", current_user)
    name      = full_name

    # ── Validation ────────────────────────────────────────────────────────────
    errors = []
    if study_time is None or study_time == "":
        errors.append("Study time is required.")
    elif float(study_time) <= 0:
        errors.append("Study time must be > 0 minutes.")
    elif float(study_time) > 600:
        errors.append("Study time cannot exceed 600 minutes.")
    if break_time is None or break_time == "":
        errors.append("Break time is required.")
    elif float(break_time) < 0:
        errors.append("Break time cannot be negative.")
    if tasks_completed is None or tasks_completed == "":
        errors.append("Tasks completed is required.")
    elif float(tasks_completed) < 0:
        errors.append("Tasks completed cannot be negative.")
    if idle_time is None or idle_time == "":
        errors.append("Distraction time is required.")
    elif float(idle_time) < 0:
        errors.append("Distraction time cannot be negative.")
    if session_time is None or session_time == "":
        errors.append("Total session duration is required.")
    elif float(session_time) < float(study_time or 0):
        errors.append("Total session duration must be >= study time.")
    if not time_of_day:
        errors.append("Please select the time of day.")
    if errors:
        raise gr.Error(" | ".join(errors))

    # ── ML Prediction ─────────────────────────────────────────────────────────
    try:
        X_sc = encode_single_input(
            int(study_time), int(break_time), int(tasks_completed),
            int(idle_time),  int(session_time), time_of_day,
            SCALER, FEATURE_COLS,
        )
        focus_score, productive, cluster_id, cluster_info = MODEL.predict(X_sc)
        cluster_name, _icon, cluster_color = cluster_info
    except Exception as exc:
        print(f"[ML ERROR] {exc}")
        raise gr.Error("Unable to generate insights. Please check your inputs and try again.")

    # ── Charts ────────────────────────────────────────────────────────────────
    try:
        fig_gauge     = score_gauge(focus_score, productive)
        fig_gantt     = session_gantt(study_time, break_time, idle_time, session_time)
        fig_breakdown = score_breakdown(
            study_time, break_time, tasks_completed,
            idle_time, session_time, focus_score,
        )
    except Exception:
        fig_gauge = fig_gantt = fig_breakdown = None

    # ── Recommendations ───────────────────────────────────────────────────────
    try:
        recs          = generate_recommendations(
            focus_score, productive, cluster_name,
            int(study_time), int(break_time), int(tasks_completed),
            int(idle_time),  int(session_time), time_of_day,
        )
        best_time_msg = best_study_time(time_of_day, focus_score)
    except Exception:
        recs          = []
        best_time_msg = "—"

    # ── Previous session stats (BEFORE saving this session) ───────────────────
    prev_df    = get_user_history(current_user)
    prev_stats = _compute_stats(prev_df)

    comparison_block = ""
    if prev_stats:
        diff   = focus_score - prev_stats["last"]
        dc     = "#16a34a" if diff > 0 else "#dc2626" if diff < 0 else "#64748b"
        d_lbl  = (f"<span style='color:{dc};font-weight:700;'>"
                  f"{'+ ' if diff > 0 else ''}{diff} pts vs last session</span>")
        best_note = " 🏆 New personal best!" if focus_score > prev_stats["best"] else ""
        comparison_block = f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                    padding:16px 20px;margin-bottom:24px;">
          <div style="font-size:10px;font-weight:700;letter-spacing:1.2px;
                      text-transform:uppercase;color:#475569;margin-bottom:10px;">
            Compared to Your Previous Sessions
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;">
            <div>
              <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">vs Last Session</div>
              <div style="font-size:15px;font-weight:700;">{d_lbl}{best_note}</div>
            </div>
            <div>
              <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">Your Best Score</div>
              <div style="font-size:15px;font-weight:700;color:#0f172a;">{prev_stats['best']}/100</div>
            </div>
            <div>
              <div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">Sessions Logged</div>
              <div style="font-size:15px;font-weight:700;color:#0f172a;">{prev_stats['count']}</div>
            </div>
          </div>
        </div>"""

    # ── Score labels ─────────────────────────────────────────────────────────
    if focus_score >= 75:
        score_color, score_label = "#16a34a", "Strong Focus"
    elif focus_score >= 50:
        score_color, score_label = "#d97706", "Moderate Focus"
    else:
        score_color, score_label = "#dc2626", "Needs Attention"

    prod_color = "#16a34a" if productive else "#dc2626"
    prod_label = "Productive Session" if productive else "Needs Improvement"
    efficiency = round((study_time / max(session_time, 1)) * 100)

    # ── AI Summary ───────────────────────────────────────────────────────────
    parts = []
    if focus_score >= 75:
        parts.append(f"Excellent session, {name}! Your focus score of <b>{focus_score}/100</b> places you among high-performing learners.")
    elif focus_score >= 50:
        parts.append(f"Good effort, {name}! Your focus score of <b>{focus_score}/100</b> shows solid progress.")
    else:
        parts.append(f"{name}, your focus score of <b>{focus_score}/100</b> indicates a challenging session — every session gives us data to improve with.")
    if idle_time and float(idle_time) > float(study_time) * 0.25:
        parts.append(f"Your distraction time ({int(idle_time)} min) is the biggest drag on your score — reducing it is your fastest win.")
    elif tasks_completed and int(tasks_completed) >= 6:
        parts.append(f"Completing <b>{int(tasks_completed)} tasks</b> shows strong execution — keep that momentum.")
    if pref_time and pref_time != time_of_day:
        parts.append(f"You prefer studying in the <b>{pref_time}</b> but this session was <b>{time_of_day}</b>.")
    if stress_level and int(stress_level) >= 7:
        parts.append(f"A stress level of <b>{int(stress_level)}/10</b> can reduce cognitive capacity.")
    if sleep_hours and float(sleep_hours) < 6.5:
        parts.append(f"Only <b>{sleep_hours} hours</b> of sleep reduces memory consolidation.")
    ai_summary = " ".join(parts)

    # ── Stat cards ───────────────────────────────────────────────────────────
    stat_cards = f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:28px;">
      <div class="stat-card">
        <div class="stat-label">Focus Score</div>
        <div class="stat-value" style="color:{score_color};">{focus_score}</div>
        <div class="stat-sub">{score_label}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Session Outcome</div>
        <div class="stat-value" style="font-size:16px;color:{prod_color};margin-top:10px;">{prod_label}</div>
        <div class="stat-sub">Based on your inputs</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Study Efficiency</div>
        <div class="stat-value" style="color:#2563eb;">{efficiency}%</div>
        <div class="stat-sub">of total session</div>
      </div>
    </div>"""

    # ── Learning style card ──────────────────────────────────────────────────
    behavior_card = f"""
    <div class="behavior-card" style="border-top:4px solid {cluster_color};
         background:linear-gradient(135deg,{cluster_color}10,{cluster_color}20);">
      <div style="font-size:10px;font-weight:700;letter-spacing:1.2px;
                  text-transform:uppercase;color:{cluster_color};margin-bottom:5px;">
        Your Learning Style
      </div>
      <div style="font-size:18px;font-weight:700;color:#0f172a;margin-bottom:8px;">
        {cluster_name}
      </div>
      <div style="font-size:13px;color:#475569;line-height:1.6;">
        Best time to study: <b style="color:#0f172a;">{best_time_msg}</b>
      </div>
    </div>"""

    # ── Recommendations ──────────────────────────────────────────────────────
    p_color = {"high": "#dc2626", "medium": "#d97706", "low": "#16a34a"}
    p_label = {"high": "High Priority", "medium": "Medium", "low": "Low"}
    rec_html = ""
    for r in recs[:5]:
        pc = p_color.get(r.priority, "#64748b")
        pl = p_label.get(r.priority, "")
        rec_html += f"""<div class="rec-item">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;">
            <span style="font-size:11px;font-weight:700;color:{pc};
                         background:{pc}14;border-radius:6px;padding:2px 9px;">{pl}</span>
            <span style="font-size:13px;font-weight:600;color:#0f172a;">{r.category}</span>
          </div>
          <div style="font-size:13.5px;color:#475569;line-height:1.65;">{r.tip}</div>
        </div>"""

    # ── Session summary table ─────────────────────────────────────────────────
    edu   = profile.get("education", "—")
    field = profile.get("field_of_study", "—")
    age   = profile.get("age", "—")

    session_rows = f"""
    <tr><td class="sum-key">Username</td><td class="sum-val">{current_user}</td>
        <td class="sum-key">Age</td><td class="sum-val">{age}</td></tr>
    <tr class="alt"><td class="sum-key">Education</td><td class="sum-val">{edu}</td>
        <td class="sum-key">Field of Study</td><td class="sum-val">{field}</td></tr>
    <tr><td class="sum-key">Study Time</td><td class="sum-val">{int(study_time)} min</td>
        <td class="sum-key">Break Time</td><td class="sum-val">{int(break_time)} min</td></tr>
    <tr class="alt"><td class="sum-key">Tasks Completed</td><td class="sum-val">{int(tasks_completed)}</td>
        <td class="sum-key">Distraction Time</td><td class="sum-val">{int(idle_time)} min</td></tr>
    <tr><td class="sum-key">Total Duration</td><td class="sum-val">{int(session_time)} min</td>
        <td class="sum-key">Time of Day</td>
        <td class="sum-val" style="text-transform:capitalize;">{time_of_day}</td></tr>"""

    # ── Build the full report ─────────────────────────────────────────────────
    report = f"""
    <div style="animation:fadeIn .4s ease;font-family:'Inter',sans-serif;">
      <div class="welcome-banner">
        <div style="font-size:22px;font-weight:800;color:#0f172a;margin-bottom:6px;">
          {"Welcome back" if prev_stats else "Welcome"},
          <span style="color:#2563eb;">{name}!</span>
        </div>
        <p style="font-size:14px;color:#475569;margin:0;line-height:1.65;">
          {"This is session #" + str(prev_stats["count"] + 1) + " for you. " if prev_stats else ""}
          Here is your personalized productivity analysis.
        </p>
      </div>
      {comparison_block}
      <div class="ai-summary-box">
        <div style="font-size:10px;font-weight:700;letter-spacing:1.2px;
                    text-transform:uppercase;color:#2563eb;margin-bottom:10px;">
          AI Insight
        </div>
        <p style="font-size:14px;color:#1e3a8a;line-height:1.75;margin:0;">{ai_summary}</p>
      </div>
      {stat_cards}
      {behavior_card}
      <div style="margin:28px 0;">
        <h3 class="section-heading">Personalized Recommendations for {name}</h3>
        <div class="rec-list">{rec_html}</div>
      </div>
      <div style="margin-top:28px;">
        <h3 class="section-heading">Session Summary</h3>
        <table class="sum-table"><tbody>{session_rows}</tbody></table>
      </div>
    </div>"""

    # Save AFTER computing prev_stats so comparison is accurate
    _save_session(
        current_user, full_name, focus_score,
        study_time, break_time, tasks_completed,
        idle_time, session_time, time_of_day, productive, cluster_name,
    )

    # Build embedded history charts
    updated_df    = get_user_history(current_user)
    display_name  = profile.get("full_name", current_user)
    hist_trend    = _trend_chart(updated_df, display_name)
    hist_bar      = _bar_chart(updated_df)
    hist_tbl_html = _history_table_html(updated_df)

    return (
        report,
        fig_gauge, fig_gantt, fig_breakdown,
        hist_trend, hist_bar, hist_tbl_html,
        gr.Tabs(selected=3),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

@keyframes fadeIn {
  from { opacity:0; transform:translateY(14px); }
  to   { opacity:1; transform:translateY(0); }
}
:root {
  --bg:#f0f4f8; --surface:#ffffff; --s2:#f8fafc;
  --border:#e2e8f0; --text:#0f172a; --text-2:#475569; --text-3:#94a3b8;
  --accent:#2563eb; --accent-d:#1d4ed8;
  --radius:16px; --shadow:0 2px 8px rgba(0,0,0,.07);
  --shadow-lg:0 8px 32px rgba(0,0,0,.10);
}
html,body,.gradio-container {
  background:var(--bg)!important;
  font-family:'Inter',system-ui,sans-serif!important;
  color:var(--text)!important;
}
.gradio-container>.main>.wrap,footer { background:transparent!important; }

/* ── Hero ────────────────────────────────────────────────────────── */
.hero-wrap {
  background:linear-gradient(135deg,#1e3a8a 0%,#2563eb 55%,#60a5fa 100%);
  border-radius:22px; padding:48px 52px 42px; margin-bottom:28px;
  box-shadow:0 12px 48px rgba(37,99,235,.30);
  position:relative; overflow:hidden;
}
.hero-wrap::before {
  content:'';position:absolute;top:-80px;right:-80px;
  width:300px;height:300px;background:rgba(255,255,255,.06);border-radius:50%;
}
.hero-wrap::after {
  content:'';position:absolute;bottom:-110px;left:22%;
  width:380px;height:380px;background:rgba(255,255,255,.04);border-radius:50%;
}
.hero-logo { display:flex;align-items:center;gap:18px;margin-bottom:18px;position:relative;z-index:1; }
.hero-mark {
  width:56px;height:56px;background:rgba(255,255,255,.18);
  border:1.5px solid rgba(255,255,255,.28);border-radius:15px;
  display:flex;align-items:center;justify-content:center;
  font-size:19px;font-weight:800;color:#fff;letter-spacing:-1px;flex-shrink:0;
}
.hero-name { font-size:32px;font-weight:800;color:#fff;letter-spacing:-1px;line-height:1.1; }
.hero-name em { color:#93c5fd;font-style:normal; }
.hero-sub { font-size:14px;color:rgba(255,255,255,.72);margin-bottom:28px;
            line-height:1.7;max-width:600px;position:relative;z-index:1; }
.hero-chips { display:flex;gap:10px;flex-wrap:wrap;position:relative;z-index:1; }
.hero-chip { background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.22);
             border-radius:9px;padding:6px 15px;font-size:12.5px;
             color:rgba(255,255,255,.90);font-weight:500; }

/* ── Page card ───────────────────────────────────────────────────── */
.page-card {
  background:var(--surface)!important; border-radius:20px!important;
  padding:38px!important; box-shadow:var(--shadow-lg)!important;
  border:1px solid var(--border)!important;
  max-width:1000px!important; margin:0 auto!important;
}

/* ── Hide tab navigation bar completely (routing is programmatic) ── */
.tab-nav,
div[role="tablist"],
.tabs > .tab-nav,
.gradio-tabs > .tab-nav,
button[role="tab"],
.tab-nav button,
.tab-nav > *,
[class*="tab-nav"],
.tabs > div:first-child > div:first-child,
.gradio-container .tabs .tab-nav { display:none!important; }

/* ── Alert system ────────────────────────────────────────────────── */
.alert { display:flex;align-items:flex-start;gap:10px;border-radius:11px;
         padding:13px 16px;font-size:13.5px;font-weight:500;
         line-height:1.6;margin-bottom:16px; }
.alert-icon { width:20px;height:20px;border-radius:50%;flex-shrink:0;
              display:flex;align-items:center;justify-content:center;
              font-size:11px;font-weight:800;color:#fff; }
.alert-error { background:#fef2f2;border:1px solid #fca5a5;color:#7f1d1d; }
.alert-error .alert-icon { background:#dc2626; }
.alert-warn  { background:#fffbeb;border:1px solid #fcd34d;color:#78350f; }
.alert-warn  .alert-icon { background:#d97706; }
.alert-ok    { background:#f0fdf4;border:1px solid #86efac;color:#14532d; }
.alert-ok    .alert-icon { background:#16a34a; }
.alert-info  { background:#eff6ff;border:1px solid #93c5fd;color:#1e3a8a; }
.alert-info  .alert-icon { background:#2563eb; }

/* ── Page headings ───────────────────────────────────────────────── */
.page-eyebrow { font-size:11px;font-weight:700;letter-spacing:1.3px;
                text-transform:uppercase;color:var(--accent);margin-bottom:6px; }
.page-title { font-size:22px;font-weight:800;color:var(--text);
              margin:0 0 8px;letter-spacing:-.4px; }
.page-desc { font-size:13.5px;color:var(--text-2);margin:0 0 28px;line-height:1.7; }
.section-divider { border:none;border-top:1px solid var(--border);margin:24px 0; }

/* ── Inputs ──────────────────────────────────────────────────────── */
label>span:first-child { color:var(--text)!important;font-size:13px!important;font-weight:600!important; }
input[type="number"],input[type="text"],input[type="email"] {
  background:var(--s2)!important;color:var(--text)!important;
  border:1.5px solid var(--border)!important;border-radius:10px!important;
  font-family:'Inter',sans-serif!important;font-size:14px!important;
  font-weight:500!important;padding:10px 14px!important;transition:border-color .15s!important;
}
input:focus {
  border-color:var(--accent)!important;
  box-shadow:0 0 0 3px rgba(37,99,235,.1)!important;outline:none!important;
}
select { background:var(--s2)!important;color:var(--text)!important;
         border:1.5px solid var(--border)!important;border-radius:10px!important;
         font-family:'Inter',sans-serif!important;font-size:14px!important;
         font-weight:500!important;padding:10px 14px!important; }

/* ── Buttons ─────────────────────────────────────────────────────── */
.btn-primary button {
  background:var(--accent)!important;color:#fff!important;
  font-family:'Inter',sans-serif!important;font-size:14.5px!important;
  font-weight:700!important;border-radius:12px!important;border:none!important;
  padding:13px 0!important;width:100%!important;
  box-shadow:0 2px 14px rgba(37,99,235,.32)!important;
  transition:all .15s!important;cursor:pointer!important;
}
.btn-primary button:hover { background:var(--accent-d)!important;
  box-shadow:0 4px 22px rgba(37,99,235,.44)!important;transform:translateY(-1px)!important; }
.btn-secondary button {
  background:var(--s2)!important;color:var(--accent)!important;
  font-family:'Inter',sans-serif!important;font-size:14px!important;
  font-weight:600!important;border-radius:12px!important;
  border:1.5px solid var(--accent)!important;padding:11px 0!important;
  width:100%!important;transition:all .15s!important;cursor:pointer!important;
}
.btn-secondary button:hover { background:#eff6ff!important; }
.btn-ghost button {
  background:transparent!important;color:var(--text-2)!important;
  font-family:'Inter',sans-serif!important;font-size:13px!important;
  font-weight:500!important;border-radius:10px!important;
  border:1.5px solid var(--border)!important;padding:10px 0!important;
  width:100%!important;transition:all .15s!important;cursor:pointer!important;
}
.btn-ghost button:hover { background:var(--s2)!important;color:var(--text)!important; }
.btn-back button {
  background:transparent!important;color:#94a3b8!important;
  font-family:'Inter',sans-serif!important;font-size:12.5px!important;
  font-weight:500!important;border-radius:10px!important;
  border:1px dashed #cbd5e1!important;padding:9px 0!important;
  width:100%!important;transition:all .15s!important;cursor:pointer!important;
  margin-top:8px!important;
}
.btn-back button:hover { background:#f8fafc!important;color:#475569!important;border-color:#94a3b8!important; }

/* ── Info banner ─────────────────────────────────────────────────── */
.info-banner { background:#eff6ff;border:1px solid #bfdbfe;border-radius:11px;
               padding:14px 18px;font-size:13px;color:#1e40af;
               margin-bottom:20px;line-height:1.6; }

/* ── Results ─────────────────────────────────────────────────────── */
.pending-msg { text-align:center;padding:60px 20px;
               color:var(--text-3);font-size:14px;font-style:italic; }
.welcome-banner { background:linear-gradient(135deg,#eff6ff,#dbeafe);
                  border:1px solid #bfdbfe;border-radius:14px;
                  padding:22px 26px;margin-bottom:24px; }
.ai-summary-box { background:#f0fdf4;border:1px solid #86efac;
                  border-left:4px solid #16a34a;border-radius:12px;
                  padding:18px 20px;margin-bottom:24px; }
.stat-card { background:var(--surface);border:1px solid var(--border);
             border-radius:14px;padding:18px;text-align:center; }
.stat-label { font-size:10px;font-weight:700;letter-spacing:1px;
              text-transform:uppercase;color:var(--text-3);margin-bottom:8px; }
.stat-value { font-size:28px;font-weight:800;color:var(--text); }
.stat-sub   { font-size:11px;color:var(--text-3);margin-top:4px; }
.behavior-card { border:1.5px solid #e2e8f0;border-radius:14px;padding:20px 22px;margin-bottom:24px; }
.section-heading { font-size:15px;font-weight:700;color:var(--text);
                   border-bottom:1px solid var(--border);padding-bottom:10px;margin-bottom:16px; }
.rec-item { padding:14px 0;border-bottom:1px solid #f1f5f9; }
.rec-item:last-child { border-bottom:none; }
.sum-table { width:100%;border-collapse:collapse;font-size:13.5px; }
.sum-table .sum-key { padding:10px 14px;border:1px solid var(--border);
                      color:var(--text-2);font-weight:600;background:var(--s2);width:22%; }
.sum-table .sum-val { padding:10px 14px;border:1px solid var(--border);
                      color:var(--text);font-weight:600;width:28%; }
.sum-table tr.alt .sum-key,.sum-table tr.alt .sum-val { background:#fff; }
.plot-card { background:var(--surface)!important;border:1px solid var(--border)!important;
             border-radius:var(--radius)!important;padding:4px!important;
             box-shadow:var(--shadow)!important; }

/* ── Embedded history section ────────────────────────────────────── */
.hist-divider {
  margin-top:40px;padding-top:32px;
  border-top:2px dashed var(--border);
}

/* ── Hide Gradio "···" share/overflow/settings button everywhere ── */
.share-button, .share-js-button, .built-with, footer,
.gradio-container footer, .overflow-menu, .extra-actions,
button[aria-label*="more"], button[aria-label*="More"],
button[aria-label*="settings"], button[aria-label*="Settings"],
button[title*="more"], .panel-options, .options-menu,
.gr-button.share-button, [title="Share"], [aria-label="Share"],
div.icon.svelte-1ed2p3z, .icon.svelte-1ed2p3z,
.show-api, button.icon { display:none!important; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border);border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--accent); }
"""


# ─────────────────────────────────────────────────────────────────────────────
# Landing HTML
# ─────────────────────────────────────────────────────────────────────────────
LANDING_HTML = """
<div style="font-family:'Inter',sans-serif;text-align:center;padding:10px 0 40px;">
  <div style="font-size:11px;font-weight:700;letter-spacing:1.4px;
              text-transform:uppercase;color:#2563eb;margin-bottom:16px;">
    AI-Powered Study Productivity Assistant
  </div>
  <h1 style="font-size:38px;font-weight:800;color:#0f172a;letter-spacing:-1.2px;margin:0 0 16px;">
    Study Smarter.<br><span style="color:#2563eb;">Grow Faster.</span>
  </h1>
  <p style="font-size:15px;color:#475569;max-width:560px;margin:0 auto 36px;line-height:1.8;">
    MindTrack AI analyses your study sessions with machine learning —
    delivering a personalised focus score, productivity rating, and an
    action plan designed around <em>how you learn</em>.
  </p>
  <div style="display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin-bottom:48px;">
    <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;
                padding:10px 20px;font-size:13px;color:#14532d;font-weight:600;">
      ✓ AI Focus Prediction
    </div>
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                padding:10px 20px;font-size:13px;color:#1e3a8a;font-weight:600;">
      ✓ History Inside Results
    </div>
    <div style="background:#fdf4ff;border:1px solid #e9d5ff;border-radius:10px;
                padding:10px 20px;font-size:13px;color:#581c87;font-weight:600;">
      ✓ Smart Recommendations
    </div>
    <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;
                padding:10px 20px;font-size:13px;color:#7c2d12;font-weight:600;">
      ✓ Learning Style Detection
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;
              max-width:720px;margin:0 auto 40px;text-align:left;">
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:22px 18px;">
      <div style="width:32px;height:32px;background:#2563eb;border-radius:50%;
                  color:#fff;font-weight:800;font-size:14px;
                  display:flex;align-items:center;justify-content:center;margin-bottom:12px;">1</div>
      <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:6px;">Create an Account</div>
      <div style="font-size:12px;color:#64748b;line-height:1.6;">
        Register with a unique username. Returning users log in instantly.
      </div>
    </div>
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:22px 18px;">
      <div style="width:32px;height:32px;background:#2563eb;border-radius:50%;
                  color:#fff;font-weight:800;font-size:14px;
                  display:flex;align-items:center;justify-content:center;margin-bottom:12px;">2</div>
      <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:6px;">Log Your Session</div>
      <div style="font-size:12px;color:#64748b;line-height:1.6;">
        Enter study time, breaks, tasks, and distractions.
      </div>
    </div>
    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:22px 18px;">
      <div style="width:32px;height:32px;background:#2563eb;border-radius:50%;
                  color:#fff;font-weight:800;font-size:14px;
                  display:flex;align-items:center;justify-content:center;margin-bottom:12px;">3</div>
      <div style="font-size:13px;font-weight:700;color:#0f172a;margin-bottom:6px;">Get Your Report + History</div>
      <div style="font-size:12px;color:#64748b;line-height:1.6;">
        AI focus score, recommendations, and your full session history — all in one place.
      </div>
    </div>
  </div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gradio App
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MindTrack AI", css=CUSTOM_CSS) as demo:

    current_user = gr.State("")  # authenticated username

    # ── Inline style injection to nuke tab bar at runtime ───────────────────
    gr.HTML("""
    <style>
      /* ── Hide tab navigation bar ── */
      .tab-nav, .tabs .tab-nav, div[role="tablist"],
      .gradio-tabitem > .tab-nav, .gradio-tabs > .tab-nav,
      .tabs > .tab-nav, ul.tab-nav,
      .tab-nav button, button[role="tab"],
      [class*="tab-nav"], [class*="tabitem"] > [class*="tab-nav"],
      .gr-tab-nav, .tabs > div > div[role="tablist"] {
        display: none !important;
        height: 0 !important;
        overflow: hidden !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
      }

      /* ── Hide Gradio overflow / share / settings "···" button ── */
      .overflow-hidden,
      .show-api,
      button.share-button,
      .gr-button.share-button,
      [aria-label="Share"],
      [title="Share"],
      .share-js-button,
      .built-with,
      .built-with-content,
      footer .built-with,
      .gradio-container > footer,
      .gradio-container footer,
      .svelte-1ed2p3z,
      button.svelte-1ed2p3z,
      div.svelte-1ed2p3z:has(button),
      .overflow-menu,
      [aria-label*="more"],
      [aria-label*="More"],
      [aria-label*="overflow"],
      [aria-label*="settings"],
      [aria-label*="Settings"],
      .extra-actions,
      .options-menu,
      button:has(svg[aria-label*="dots"]),
      button[title*="more"],
      .gr-form > button:last-of-type:not([class]),
      .panel > .options,
      .panel-options,
      div.panel > div:last-child > button.icon,
      .wrap > .options,
      .fixed-height > button.icon,
      #component-0 > div.wrap > button,
      .app > div > button.icon,
      footer { display:none!important; visibility:hidden!important; }
    </style>""")

    # ── Global hero header ────────────────────────────────────────────────────
    gr.HTML("""
    <div class='hero-wrap'>
      <div class='hero-logo'>
        <div class='hero-mark'>MT</div>
        <div class='hero-name'>MindTrack <em>AI</em></div>
      </div>
      <p class='hero-sub'>
        Your personal AI study coach — log a session and get a personalised
        focus report with embedded history tracking.
      </p>
      <div class='hero-chips'>
        <div class='hero-chip'>Unique User Accounts</div>
        <div class='hero-chip'>AI Focus Prediction</div>
        <div class='hero-chip'>History Inside Results</div>
        <div class='hero-chip'>Smart Recommendations</div>
      </div>
    </div>""")

    with gr.Column(elem_classes=["page-card"]):
        with gr.Tabs() as main_tabs:

            # ══════════════════════════════════════════════════════════════════
            # PAGE 0 — LANDING
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Landing", id=0):
                gr.HTML(LANDING_HTML)
                with gr.Row():
                    with gr.Column(elem_classes=["btn-primary"]):
                        btn_go_login    = gr.Button("Login", variant="primary")
                    with gr.Column(elem_classes=["btn-secondary"]):
                        btn_go_register = gr.Button("Register", variant="secondary")

            # ══════════════════════════════════════════════════════════════════
            # PAGE 1 — LOGIN
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Login", id=1):
                gr.HTML("""
                <div class='page-eyebrow'>Welcome Back</div>
                <h2 class='page-title'>Log In to Your Account</h2>
                <p class='page-desc'>
                  Enter your username and password to access your personalised dashboard.
                </p>""")

                login_alert    = gr.HTML(value="", visible=False)
                inp_login_user = gr.Textbox(
                    label="Username / Email",
                    placeholder="e.g. rania_asad",
                    info="Your unique username — the same one you registered with.")
                inp_login_pass = gr.Textbox(
                    label="Password",
                    placeholder="Enter your password",
                    type="password",
                    info="Your account password.")

                with gr.Row(elem_classes=["btn-primary"]):
                    btn_login = gr.Button("Log In →", variant="primary")

                gr.HTML("""<div style="text-align:center;margin-top:24px;padding-top:20px;
                    border-top:1px solid #e2e8f0;font-size:13px;color:#64748b;">
                  Don't have an account?
                </div>""")
                with gr.Row(elem_classes=["btn-ghost"]):
                    btn_switch_to_register = gr.Button("Create an Account →")

                with gr.Row(elem_classes=["btn-back"]):
                    btn_login_back_landing = gr.Button("← Back to Home")

            # ══════════════════════════════════════════════════════════════════
            # PAGE 5 — REGISTER
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Register", id=5):
                gr.HTML("""
                <div class='page-eyebrow'>New User</div>
                <h2 class='page-title'>Create Your Account</h2>
                <p class='page-desc'>
                  Fill in the details below. You'll be redirected to login once registered.
                </p>""")

                reg_alert = gr.HTML(value="", visible=False)

                with gr.Row():
                    reg_username = gr.Textbox(
                        label="Username *",
                        placeholder="e.g. rania_asad  (letters, numbers, ._-)",
                        info="Choose a unique username — you'll use this to log in every time.")
                    reg_name = gr.Textbox(
                        label="Full Name *",
                        placeholder="e.g. Rania Asad")

                with gr.Row():
                    reg_age    = gr.Number(value=21, minimum=13, maximum=100, label="Age *")
                    reg_gender = gr.Radio(
                        choices=["Male", "Female", "Prefer not to say"],
                        value="Female", label="Gender *")

                with gr.Row():
                    reg_education = gr.Dropdown(
                        choices=["High School", "Diploma", "Bachelor's Degree",
                                 "Master's Degree", "PhD / Doctorate", "Other"],
                        value="Bachelor's Degree",
                        label="Education Level *")
                    reg_field = gr.Textbox(
                        label="Field of Study",
                        placeholder="e.g. Computer Science, Medicine  (optional)")

                with gr.Row():
                    reg_password = gr.Textbox(
                        label="Password *",
                        placeholder="Minimum 6 characters",
                        type="password",
                        info="Choose a secure password — at least 6 characters.")
                    reg_confirm_pass = gr.Textbox(
                        label="Confirm Password *",
                        placeholder="Re-enter your password",
                        type="password",
                        info="Must match the password above.")

                with gr.Row(elem_classes=["btn-primary"]):
                    btn_register = gr.Button("Create Account & Continue", variant="primary")

                gr.HTML("""<div style="text-align:center;margin-top:24px;padding-top:20px;
                    border-top:1px solid #e2e8f0;font-size:13px;color:#64748b;">
                  Already have an account?
                </div>""")
                with gr.Row(elem_classes=["btn-ghost"]):
                    btn_switch_to_login = gr.Button("← Back to Login")

                with gr.Row(elem_classes=["btn-back"]):
                    btn_reg_back_landing = gr.Button("← Back to Home")

            # ══════════════════════════════════════════════════════════════════
            # PAGE 2 — INPUT SECTION
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Input", id=2):
                gr.HTML("""
                <div class='page-eyebrow'>Session Details</div>
                <h2 class='page-title'>Log Your Study Session</h2>
                <p class='page-desc'>
                  Enter the details of your study session. All starred fields are required.
                  Approximate values are fine — the AI will work with what you provide.
                </p>""")

                with gr.Row():
                    inp_study   = gr.Number(value=90,  minimum=1,  maximum=600,
                        label="Study Time (minutes) *",
                        info="Time actively reading, writing, or solving problems.")
                    inp_break   = gr.Number(value=20,  minimum=0,  maximum=300,
                        label="Break Time (minutes) *",
                        info="Planned breaks — stepping away to rest.")

                with gr.Row():
                    inp_tasks   = gr.Number(value=5,   minimum=0,  maximum=100,
                        label="Tasks Completed *",
                        info="Number of tasks, exercises, or problems finished.")
                    inp_idle    = gr.Number(value=15,  minimum=0,  maximum=300,
                        label="Distraction Time (minutes) *",
                        info="Time lost to phone, social media, or daydreaming.")

                with gr.Row():
                    inp_session = gr.Number(value=130, minimum=5,  maximum=720,
                        label="Total Session Duration (minutes) *",
                        info="From when you sat down to when you stopped.")
                    inp_tod     = gr.Dropdown(
                        choices=["morning", "afternoon", "evening", "night"],
                        value="morning",
                        label="Time of Day *",
                        info="Which part of the day did this session take place?")

                gr.HTML("<hr class='section-divider'>")
                gr.HTML("""<div class='page-eyebrow' style='margin-top:4px;'>
                  Optional — helps personalise your AI insights
                </div>""")

                with gr.Row():
                    inp_goal      = gr.Number(value=4, minimum=1, maximum=16,
                        label="Daily Study Goal (hours)")
                    inp_pref_time = gr.Radio(
                        choices=["morning", "afternoon", "evening", "night"],
                        value="morning", label="Preferred Study Time")

                with gr.Row():
                    inp_sleep  = gr.Number(value=7, minimum=3, maximum=12,
                        label="Average Sleep Hours")
                    inp_stress = gr.Number(value=4, minimum=1, maximum=10,
                        label="Stress Level  (1 = Calm,  10 = Very Stressed)")

                gr.HTML("""<div class='info-banner' style='margin-top:16px;'>
                  Click <b>Analyse My Session</b> to generate your personalised report
                  and update your history.
                </div>""")

                with gr.Row(elem_classes=["btn-primary"]):
                    btn_analyze = gr.Button("Analyse My Session →", variant="primary")

            # ══════════════════════════════════════════════════════════════════
            # PAGE 3 — RESULTS  (with embedded history)
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("Results", id=3):
                gr.HTML("""
                <div class='page-eyebrow'>Productivity Dashboard</div>
                <h2 class='page-title'>Your Personalised Results</h2>""")

                result_html = gr.HTML(
                    value="<div class='pending-msg'>Log in, enter session details, "
                          "then click Analyse My Session.</div>")

                with gr.Row():
                    gauge_plot = gr.Plot(label="Focus Score",       elem_classes=["plot-card"])
                    gantt_plot = gr.Plot(label="Session Timeline",  elem_classes=["plot-card"])

                breakdown_plot = gr.Plot(label="Score Breakdown",   elem_classes=["plot-card"])

                # ── Action buttons ────────────────────────────────────────────
                gr.HTML("<hr class='section-divider'>")
                with gr.Row():
                    with gr.Column(elem_classes=["btn-secondary"]):
                        btn_new_session = gr.Button("← Analyse Another Session", variant="secondary")
                    with gr.Column(elem_classes=["btn-ghost"]):
                        btn_logout      = gr.Button("Log Out")

                # ── Embedded History section ───────────────────────────────────
                gr.HTML("""
                <div style="margin-top:40px;padding-top:32px;border-top:2px dashed #e2e8f0;">
                  <div style="font-size:11px;font-weight:700;letter-spacing:1.3px;
                              text-transform:uppercase;color:#2563eb;margin-bottom:8px;">
                    Your Progress Over Time
                  </div>
                  <h3 style="font-size:18px;font-weight:800;color:#0f172a;
                             margin:0 0 6px;letter-spacing:-.3px;">
                    Session History
                  </h3>
                  <p style="font-size:13px;color:#475569;margin:0 0 24px;line-height:1.7;">
                    Updated automatically after each analysis. Your records are private and
                    never mixed with other users.
                  </p>
                </div>""")

                with gr.Row():
                    hist_trend_plot = gr.Plot(label="Focus Score Trend",            elem_classes=["plot-card"])
                    hist_bar_plot   = gr.Plot(label="Session-by-Session Breakdown", elem_classes=["plot-card"])

                hist_table_html = gr.HTML(value="")

    gr.HTML("""
    <div style='text-align:center;padding:22px 0 10px;color:#94a3b8;
                font-size:12.5px;border-top:1px solid #e2e8f0;margin-top:28px;'>
      MindTrack AI &nbsp;·&nbsp; AI-Powered Study Productivity Assistant
      &nbsp;|&nbsp; Powered by scikit-learn and Gradio
    </div>""")

    # ── Navigation wiring ─────────────────────────────────────────────────────

    # Landing → Login / Register
    btn_go_login.click(fn=lambda: gr.Tabs(selected=1),    outputs=main_tabs)
    btn_go_register.click(fn=lambda: gr.Tabs(selected=5), outputs=main_tabs)

    # Switch between Login ↔ Register forms
    btn_switch_to_register.click(fn=lambda: gr.Tabs(selected=5), outputs=main_tabs)
    btn_switch_to_login.click(fn=lambda: gr.Tabs(selected=1),    outputs=main_tabs)

    # Back to Landing from Login / Register
    btn_login_back_landing.click(fn=lambda: gr.Tabs(selected=0), outputs=main_tabs)
    btn_reg_back_landing.click(fn=lambda: gr.Tabs(selected=0),   outputs=main_tabs)

    # Login → Input
    btn_login.click(
        fn=handle_login,
        inputs=[inp_login_user, inp_login_pass],
        outputs=[login_alert, main_tabs, current_user],
    )

    # Register → Login (success shown on login page via reg_alert)
    btn_register.click(
        fn=handle_register,
        inputs=[reg_username, reg_name, reg_age, reg_gender, reg_education, reg_field,
                reg_password, reg_confirm_pass],
        outputs=[reg_alert, main_tabs],
    )

    # Analyse → Results (with embedded history outputs)
    btn_analyze.click(
        fn=run_analysis,
        inputs=[
            current_user,
            inp_study, inp_break, inp_tasks,
            inp_idle,  inp_session, inp_tod,
            inp_goal,  inp_pref_time, inp_sleep, inp_stress,
        ],
        outputs=[
            result_html,
            gauge_plot, gantt_plot, breakdown_plot,
            hist_trend_plot, hist_bar_plot, hist_table_html,
            main_tabs,
        ],
    )

    # Results → Input (new session)
    btn_new_session.click(fn=lambda: gr.Tabs(selected=2), outputs=main_tabs)

    # Logout → Landing (clear state)
    btn_logout.click(
        fn=lambda: ("", gr.Tabs(selected=0)),
        outputs=[current_user, main_tabs],
    )


if __name__ == "__main__":
    demo.launch(share=False, show_error=True)
