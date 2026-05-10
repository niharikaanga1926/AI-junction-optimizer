"""
features_engine.py — AI Junction Optimizer: 5-Feature Logic Engine
Reads from existing controller/detector state — never duplicates it.
"""

import asyncio
import random
import time
import httpx
from datetime import datetime
from typing import Optional
from collections import deque
from dotenv import load_dotenv
import os
load_dotenv()
# ─────────────────────────────────────────────────────────────
# Shared references (set by features_router on startup)
# ─────────────────────────────────────────────────────────────

_controller = None   # traffic_signal.controller.SignalController
_density_history = None  # collections.deque from main.py
_session_log = None      # list from main.py

ARM_NAMES = ["North", "South", "East", "West"]

# ─────────────────────────────────────────────────────────────
# Global Feature State
# ─────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY", "")
# Feature 1 – Lane suggestions
lane_suggestions: list[dict] = []

# Feature 2 – AI Briefing
ai_briefing: dict = {
    "text": "Awaiting first analysis…",
    "timestamp": None,
    "next_refresh_in": 30,
    "scenario": None,
}
_last_briefing_ts: float = 0.0

# Feature 3 – Efficiency panel
FIXED_BASELINE = {
    "avg_wait_s": 45.0,
    "throughput_vpm": 20.0,
    "co2_g_per_min": 100.0,
}

# Feature 4 – Simulation mode
simulation: dict = {
    "active": False,
    "scenario": None,
    "started_at": None,
    "ends_at": None,
    "overrides": {},   # arm -> density override
}
_sim_incidents: list[dict] = []

# Feature 5 – Carbon savings
carbon: dict = {
    "total_kg_saved": 0.0,
    "rate_g_per_min": 0.0,
    "history": deque(maxlen=120),   # up to 2 min of sparkline data
}
_last_carbon_ts: float = time.time()


# ─────────────────────────────────────────────────────────────
# Initialiser
# ─────────────────────────────────────────────────────────────

def init_engine(controller, density_history, session_log):
    global _controller, _density_history, _session_log
    _controller = controller
    _density_history = density_history
    _session_log = session_log


# ─────────────────────────────────────────────────────────────
# Feature 1 – Lane Management Suggestions
# ─────────────────────────────────────────────────────────────

def compute_lane_suggestions() -> list[dict]:
    """
    Reads controller.density (set externally by detector).
    Triggers when any direction density > 50%.
    Returns list of advisory dicts; also appends to lane_suggestions.
    """
    global lane_suggestions

    if _controller is None:
        return lane_suggestions

    new_suggestions = []
    densities = dict(_controller.density)

    for arm, density in densities.items():
        if density > 50.0:
            # Decide green extension based on severity
            extend_s = int(round((density - 50) / 5)) * 2 + 4   # 4-20s
            msg = (
                f"{arm} {density:.0f}% → Activate overflow lane | "
                f"Extend green +{extend_s}s"
            )
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "arm": arm,
                "density": density,
                "message": msg,
                "extend_green_s": extend_s,
                "type": "LANE_ADVISORY",
            }
            new_suggestions.append(entry)

            # Log to controller incidents (append only — no modification)
            if _controller and hasattr(_controller, "_incidents"):
                _controller._incidents.append({
                    "timestamp": entry["timestamp"],
                    "message": msg,
                    "type": "LANE_ADVISORY",
                    "arm": arm,
                })

    # Keep only last 20; prioritise fresh entries
    if new_suggestions:
        lane_suggestions = (lane_suggestions + new_suggestions)[-20:]

    return lane_suggestions


# ─────────────────────────────────────────────────────────────
# Feature 2 – AI Natural Language Briefing (Claude API)
# ─────────────────────────────────────────────────────────────

async def refresh_ai_briefing() -> dict:
    """
    Calls claude-sonnet-4-20250514 every 30s.
    Reads live density + signal state + KPIs from existing objects.
    Returns updated ai_briefing dict.
    """
    global ai_briefing, _last_briefing_ts

    now = time.time()
    elapsed = now - _last_briefing_ts
    ai_briefing["next_refresh_in"] = max(0, int(30 - elapsed))

    if elapsed < 30:
        return ai_briefing

    if not ANTHROPIC_API_KEY:
        ai_briefing["text"] = "⚠ No API key set. POST /api/features/apikey to enable AI briefings."
        ai_briefing["timestamp"] = datetime.utcnow().isoformat()
        _last_briefing_ts = now
        ai_briefing["next_refresh_in"] = 30
        return ai_briefing

    # Build context from existing state
    densities = dict(_controller.density) if _controller else {}
    signals = {a: _controller.signals[a].value for a in ARM_NAMES} if _controller else {}
    wait_time = _compute_avg_wait()
    throughput = _compute_throughput()
    sim_scenario = simulation.get("scenario") if simulation["active"] else None

    prompt_parts = [
        "You are a traffic operations AI. Provide a 2-line briefing (max 200 chars total).",
        f"Junction densities (%): {', '.join(f'{a}={densities.get(a,0):.0f}' for a in ARM_NAMES)}.",
     "Signal states: " + ", ".join([a + "=" + signals.get(a, "red") for a in ARM_NAMES]) + ".",
        f"Avg wait: {wait_time:.1f}s | Throughput: {throughput:.1f} v/min.",
    ]
    if sim_scenario:
        prompt_parts.append(f"Active simulation scenario: {sim_scenario}.")
    prompt_parts.append("Return exactly 2 lines: line 1 = situation summary, line 2 = recommended action.")

    prompt = " ".join(prompt_parts)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 120,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            body = resp.json()
            text = body["content"][0]["text"].strip()
    except Exception as exc:
        text = f"Briefing unavailable: {exc}"

    _last_briefing_ts = time.time()
    ai_briefing.update({
        "text": text,
        "timestamp": datetime.utcnow().isoformat(),
        "next_refresh_in": 30,
        "scenario": sim_scenario,
    })
    return ai_briefing


# ─────────────────────────────────────────────────────────────
# Feature 3 – Efficiency Panel (Before vs After)
# ─────────────────────────────────────────────────────────────

def compute_efficiency() -> dict:
    """
    Compares fixed baseline vs live AI KPIs.
    Returns % improvements and an AI Impact Score.
    """
    ai_wait = _compute_avg_wait()
    ai_throughput = _compute_throughput()
    ai_co2 = _compute_co2()

    def pct_improvement(baseline, ai_val, lower_is_better=True):
        if baseline == 0:
            return 0.0
        if lower_is_better:
            return round((baseline - ai_val) / baseline * 100, 1)
        else:
            return round((ai_val - baseline) / baseline * 100, 1)

    wait_imp = pct_improvement(FIXED_BASELINE["avg_wait_s"], ai_wait, lower_is_better=True)
    thru_imp = pct_improvement(FIXED_BASELINE["throughput_vpm"], ai_throughput, lower_is_better=False)
    co2_imp = pct_improvement(FIXED_BASELINE["co2_g_per_min"], ai_co2, lower_is_better=True)

    # AI Impact Score: weighted average of improvements, clamped 0-100
    impact_score = max(0.0, min(100.0, (wait_imp * 0.4 + thru_imp * 0.35 + co2_imp * 0.25)))

    return {
        "baseline": FIXED_BASELINE,
        "ai": {
            "avg_wait_s": round(ai_wait, 1),
            "throughput_vpm": round(ai_throughput, 1),
            "co2_g_per_min": round(ai_co2, 1),
        },
        "improvements": {
            "wait_pct": wait_imp,
            "throughput_pct": thru_imp,
            "co2_pct": co2_imp,
        },
        "ai_impact_score": round(impact_score, 1),
    }


# ─────────────────────────────────────────────────────────────
# Feature 4 – Simulation Mode
# ─────────────────────────────────────────────────────────────

SCENARIOS = {
    "rush_hour":   {"label": "🚗 Rush Hour",        "duration": 30},
    "emergency":   {"label": "🚨 Emergency Vehicle", "duration": 30},
    "accident":    {"label": "⚠️  Accident",          "duration": 30},
    "off_peak":    {"label": "🌙 Off-Peak",           "duration": 30},
}


def start_simulation(scenario_key: str) -> dict:
    """
    Applies density overrides to the controller.
    Does NOT modify controller internals — only calls update_density().
    """
    global simulation

    if scenario_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_key}. Choose from {list(SCENARIOS)}")

    now = time.time()
    meta = SCENARIOS[scenario_key]
    overrides = _build_overrides(scenario_key)

    simulation.update({
        "active": True,
        "scenario": scenario_key,
        "label": meta["label"],
        "started_at": now,
        "ends_at": now + meta["duration"],
        "overrides": overrides,
    })

    # Apply overrides immediately
    _apply_sim_overrides()

    # Log to sim_incidents
    _sim_incidents.append({
        "timestamp": datetime.utcnow().isoformat(),
        "type": "SIM",
        "message": f"Simulation started: {meta['label']}",
        "scenario": scenario_key,
    })

    # Also push to controller incidents list (append-only)
    if _controller:
        _controller._incidents.append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"[SIM] {meta['label']} scenario activated",
            "type": "SIM",
        })

    return simulation


def stop_simulation() -> dict:
    """Auto-called after duration; can also be called manually."""
    global simulation

    if not simulation["active"]:
        return simulation

    scenario_key = simulation.get("scenario", "unknown")
    label = simulation.get("label", scenario_key)

    simulation.update({
        "active": False,
        "scenario": None,
        "label": None,
        "started_at": None,
        "ends_at": None,
        "overrides": {},
    })

    _sim_incidents.append({
        "timestamp": datetime.utcnow().isoformat(),
        "type": "SIM",
        "message": f"Simulation ended: {label}",
        "scenario": scenario_key,
    })

    if _controller:
        _controller._incidents.append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"[SIM] {label} scenario ended — resuming live data",
            "type": "SIM",
        })

    return simulation


def tick_simulation():
    """Call every second; auto-stops after duration."""
    if not simulation["active"]:
        return
    if time.time() >= simulation["ends_at"]:
        stop_simulation()
        return
    _apply_sim_overrides()


def _build_overrides(scenario_key: str) -> dict:
    """Return {arm: density} override dict for a scenario."""
    if scenario_key == "rush_hour":
        return {arm: random.uniform(70, 90) for arm in ARM_NAMES}

    elif scenario_key == "emergency":
        em_arm = random.choice(ARM_NAMES)
        overrides = {arm: random.uniform(20, 45) for arm in ARM_NAMES}
        overrides[em_arm] = 95.0
        overrides["_emergency_arm"] = em_arm   # sentinel for WS payload
        return overrides

    elif scenario_key == "accident":
        locked_arm = random.choice(ARM_NAMES)
        overrides = {arm: random.uniform(30, 55) for arm in ARM_NAMES}
        overrides[locked_arm] = 100.0
        overrides["_locked_arm"] = locked_arm
        return overrides

    elif scenario_key == "off_peak":
        return {arm: random.uniform(5, 15) for arm in ARM_NAMES}

    return {}


def _apply_sim_overrides():
    """Push current overrides into controller.density via update_density()."""
    if _controller is None or not simulation["active"]:
        return
    for arm in ARM_NAMES:
        val = simulation["overrides"].get(arm)
        if val is not None:
            _controller.update_density(arm, float(val))

    # Handle emergency arm trigger for emergency scenario
    em_arm = simulation["overrides"].get("_emergency_arm")
    if em_arm and not _controller.emergency_active:
        _controller.trigger_emergency(em_arm)


# ─────────────────────────────────────────────────────────────
# Feature 5 – Carbon Savings Tracker
# ─────────────────────────────────────────────────────────────

CO2_FIXED_G_PER_SEC = FIXED_BASELINE["co2_g_per_min"] / 60.0   # baseline rate

def tick_carbon():
    """
    Accumulates CO2 saved vs fixed timer every second.
    Call once per second from the WS broadcast loop.
    """
    global _last_carbon_ts

    now = time.time()
    elapsed = now - _last_carbon_ts
    if elapsed < 0.5:
        return

    ai_co2_g_per_min = _compute_co2()
    ai_g_per_sec = ai_co2_g_per_min / 60.0

    saved_g = max(0.0, (CO2_FIXED_G_PER_SEC - ai_g_per_sec) * elapsed)
    carbon["total_kg_saved"] = round(carbon["total_kg_saved"] + saved_g / 1000.0, 4)
    carbon["rate_g_per_min"] = round(max(0.0, CO2_FIXED_G_PER_SEC * 60 - ai_co2_g_per_min), 2)
    carbon["history"].append(round(carbon["total_kg_saved"], 4))

    _last_carbon_ts = now


def get_carbon_snapshot() -> dict:
    return {
        "total_kg_saved": carbon["total_kg_saved"],
        "rate_g_per_min": carbon["rate_g_per_min"],
        "history": list(carbon["history"]),
    }


# ─────────────────────────────────────────────────────────────
# Internal KPI Helpers  (read from existing state, no duplication)
# ─────────────────────────────────────────────────────────────

def _compute_avg_wait() -> float:
    if _controller is None:
        return FIXED_BASELINE["avg_wait_s"]
    avg_density = sum(_controller.density.values()) / max(1, len(ARM_NAMES))
    # Mirrors kpis() logic in main.py: wait ≈ density * 0.9
    return round(avg_density * 0.9, 1)


def _compute_throughput() -> float:
    if not _density_history:
        return FIXED_BASELINE["throughput_vpm"]
    recent = list(_density_history)[-30:]
    total_vehicles = sum(
        sum(s.get("arms", {}).get(a, {}).get("vehicle_count", 0) for a in ARM_NAMES)
        for s in recent
    )
    # vehicles per minute — mirrors kpis() in main.py
    return round(total_vehicles / max(1, len(recent) * 2 / 60), 1)


def _compute_co2() -> float:
    """CO2 g/min from idle vehicles (mirrors main.py kpis())."""
    if _controller is None or not _density_history:
        return FIXED_BASELINE["co2_g_per_min"]
    CO2_PER_VEHICLE_G_PER_MIN = 14.0
    last = _density_history[-1] if _density_history else {}
    idle = sum(
        last.get("arms", {}).get(a, {}).get("vehicle_count", 0)
        for a in ARM_NAMES
        if _controller.signals.get(a) and _controller.signals[a].value != "green"
    )
    return round(idle * CO2_PER_VEHICLE_G_PER_MIN, 1)


def get_sim_incidents() -> list:
    return _sim_incidents[-50:]
