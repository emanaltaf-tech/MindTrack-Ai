"""
recommendations.py
------------------
Generates smart, context-aware study recommendations based on:
  - Predicted focus score
  - Productivity label
  - Cluster (behaviour type)
  - Raw input values
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Recommendation:
    category: str       # e.g. "Study Time", "Break Strategy"
    icon: str
    tip: str
    priority: str       # "high" | "medium" | "low"


def generate_recommendations(
    focus_score: int,
    productive: bool,
    cluster_name: str,
    study_time: int,
    break_time: int,
    tasks_completed: int,
    idle_time: int,
    session_time: int,
    time_of_day: str,
) -> List[Recommendation]:
    """Return a prioritised list of actionable recommendations."""

    recs: List[Recommendation] = []

    # ── 1. Focus Score tier ───────────────────────────────────────────────
    if focus_score < 40:
        recs.append(Recommendation(
            "Focus Level", "🚨",
            "Your focus is critically low. Start with just ONE task and close all unrelated tabs.",
            "high",
        ))
    elif focus_score < 60:
        recs.append(Recommendation(
            "Focus Level", "📈",
            "You're getting there! Use the 2-minute rule: if a task takes <2 min, do it immediately.",
            "medium",
        ))
    elif focus_score < 80:
        recs.append(Recommendation(
            "Focus Level", "✅",
            "Good focus! Protect this momentum with a consistent pre-study ritual (same music/desk/time).",
            "low",
        ))
    else:
        recs.append(Recommendation(
            "Focus Level", "🏆",
            "Peak performance! You're in flow. Document what worked today and replicate it tomorrow.",
            "low",
        ))

    # ── 2. Idle time ──────────────────────────────────────────────────────
    idle_ratio = idle_time / max(session_time, 1)
    if idle_ratio > 0.30:
        recs.append(Recommendation(
            "Idle Time", "⏰",
            f"You spent {idle_time} min idle ({idle_ratio:.0%} of your session). "
            "Try website blockers (Cold Turkey / Freedom) during study blocks.",
            "high",
        ))
    elif idle_ratio > 0.15:
        recs.append(Recommendation(
            "Idle Time", "📵",
            "Moderate idle time detected. Keep your phone in another room during study blocks.",
            "medium",
        ))

    # ── 3. Break strategy ────────────────────────────────────────────────
    break_ratio = break_time / max(study_time, 1)
    if break_ratio > 0.50:
        recs.append(Recommendation(
            "Break Strategy", "☕",
            f"Breaks ({break_time} min) exceed 50% of study time. "
            "Switch to Pomodoro: 25 min study → 5 min break → repeat × 4 → 20 min long break.",
            "high",
        ))
    elif break_ratio < 0.08 and study_time > 60:
        recs.append(Recommendation(
            "Break Strategy", "🧘",
            "You're barely taking breaks! Short breaks every 45–50 min prevent mental fatigue and improve retention by 20%.",
            "medium",
        ))
    else:
        recs.append(Recommendation(
            "Break Strategy", "✔️",
            "Your break ratio looks healthy. Spend breaks walking or stretching — avoid scrolling social media.",
            "low",
        ))

    # ── 4. Study duration ────────────────────────────────────────────────
    if study_time < 30:
        recs.append(Recommendation(
            "Study Duration", "📅",
            "Under 30 min of study is rarely enough for deep work. Aim for at least 2–3 focused blocks per day.",
            "high",
        ))
    elif study_time > 240:
        recs.append(Recommendation(
            "Study Duration", "🛑",
            "Studying over 4 hours non-stop causes diminishing returns. Split into 2 sessions with a proper meal break.",
            "medium",
        ))

    # ── 5. Time-of-day personalisation ───────────────────────────────────
    tod_tips = {
        "morning": ("🌅", "Morning sessions are optimal — your prefrontal cortex is freshest. Tackle your hardest material first."),
        "afternoon": ("☀️", "Afternoon dip is real (2–3 PM). Schedule lighter review tasks then, save hard problems for earlier or later."),
        "evening": ("🌆", "Evening studying is fine, but stop screens 1 hour before bed. Blue light disrupts melatonin and memory consolidation."),
        "night": ("🌙", "Night studying works for some, but sleep converts short-term memory to long-term. Prioritise 7–8 hours after any night session."),
    }
    icon, tip = tod_tips.get(time_of_day, ("🕐", "Consistent study timing trains your brain's focus response."))
    recs.append(Recommendation("Best Study Time", icon, tip, "medium"))

    # ── 6. Tasks productivity ────────────────────────────────────────────
    if tasks_completed == 0:
        recs.append(Recommendation(
            "Task Management", "📋",
            "Zero tasks completed. Before your next session, write exactly 3 concrete tasks on paper. Start with the easiest.",
            "high",
        ))
    elif tasks_completed <= 3:
        recs.append(Recommendation(
            "Task Management", "📝",
            "Try task-batching: group similar tasks together (all reading → all writing → all practice problems).",
            "medium",
        ))
    else:
        recs.append(Recommendation(
            "Task Management", "🎯",
            f"Great — {tasks_completed} tasks completed! Use weekly reviews (Sunday evening, 15 min) to plan the week ahead.",
            "low",
        ))

    # ── 7. Behaviour-cluster specific ───────────────────────────────────
    cluster_tips = {
        "Deep Focus Worker":
            ("🧠", "high", "Leverage your focus strength: try 90-min deep-work blocks (ultradian rhythm). Protect your mornings fiercely."),
        "Night Owl Grinder":
            ("🌙", "medium", "Night work is your edge. Make sure tasks are planned before the session starts — late-night planning wastes your peak hours."),
        "Casual Learner":
            ("📚", "medium", "Build momentum with streaks. Even 45 min daily beats irregular 5-hour binges. Use a habit tracker app."),
        "Distracted Drifter":
            ("⚡", "high", "Your idle time is the bottleneck. Try a physical 'distraction notepad' — write distractions down instead of acting on them during study."),
    }
    for cname, (icon, pri, tip) in cluster_tips.items():
        if cname.lower() in cluster_name.lower():
            recs.append(Recommendation("Behaviour Insight", icon, tip, pri))
            break

    # Sort: high priority first
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: priority_order[r.priority])

    return recs


def best_study_time(time_of_day: str, focus_score: int) -> str:
    """Return a natural-language recommendation for optimal study time."""
    if focus_score >= 70:
        return f"Keep studying in the <b>{time_of_day}</b> — it's clearly working for you!"
    if focus_score < 50:
        if time_of_day in ("evening", "night"):
            return "Consider shifting to <b>morning</b> sessions when cognitive performance peaks for most people."
        return "Try <b>morning</b> (8–10 AM) for hard topics, <b>afternoon</b> (4–6 PM) for review."
    return "Your current time works. Experiment with starting 30 min earlier to ease into focus."
