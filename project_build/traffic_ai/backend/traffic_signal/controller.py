"""
Step 2: Signal Control Simulation
4-phase traffic light state machine with adaptive green timing,
emergency override, and lane reversal recommendation logic.
"""

import time
import threading
from datetime import datetime
from typing import Optional
from enum import Enum

# ── Types ─────────────────────────────────────────────────────────────────────

class SignalState(Enum):
    GREEN  = "green"
    YELLOW = "yellow"
    RED    = "red"
    EMERGENCY_OVERRIDE = "emergency_override"


ARM_NAMES = ["North", "South", "East", "West"]

# Timing constants (seconds)
MIN_GREEN     = 15
MAX_GREEN     = 90
YELLOW_PHASE  = 4
ALL_RED_PHASE = 2          # safety clearance between phases
EMERGENCY_GREEN = 30       # green duration for emergency arm
EMERGENCY_HOLD  = 30       # total hold time during override


class PhaseTimer:
    """Counts down from `duration` seconds."""

    def __init__(self, duration: float):
        self.duration  = duration
        self.started   = time.time()

    @property
    def remaining(self) -> float:
        return max(0.0, self.duration - (time.time() - self.started))

    @property
    def expired(self) -> bool:
        return self.remaining <= 0


# ── Signal Controller ─────────────────────────────────────────────────────────

class SignalController:
    """
    Manages a 4-arm traffic junction signal.

    Cycle:
      Arm[i] GREEN (adaptive) → YELLOW (4s) → ALL RED (2s) → next arm
    Emergency override:
      All arms RED → emergency arm GREEN (30s) → resume normal cycle
    """

    def __init__(self, on_state_change=None):
        """
        on_state_change: callable(state_dict) pushed every tick.
        """
        self.on_state_change = on_state_change

        # Current signal state per arm
        self.signals: dict[str, SignalState] = {a: SignalState.RED for a in ARM_NAMES}
        self.current_arm_index = 0       # which arm is currently active
        self.phase_timer: Optional[PhaseTimer] = None
        self.in_yellow = False
        self.in_all_red = False
        self.emergency_active = False
        self.emergency_arm: Optional[str] = None

        # Density scores updated externally
        self.density: dict[str, float] = {a: 0.0 for a in ARM_NAMES}
        # Cycle counter for lane reversal logic
        self.cycle_counts: dict[str, int] = {a: 0 for a in ARM_NAMES}
        self.lane_reversal_suggestions: list[dict] = []

        self.running = False
        self._lock = threading.Lock()
        self._incidents: list[dict] = []

        # Green phase history (for RL training data export)
        self.phase_log: list[dict] = []

    # ── External API ──────────────────────────────────────────────────────

    def update_density(self, arm: str, score: float):
        """Called by detector every 2s with new density score."""
        with self._lock:
            self.density[arm] = max(0.0, min(100.0, score))

    def trigger_emergency(self, arm: str):
        """Called when emergency vehicle detected on `arm`."""
        with self._lock:
            if not self.emergency_active:
                self.emergency_active = True
                self.emergency_arm = arm
                self._log(f"EMERGENCY override for arm {arm}")

    def clear_emergency(self):
        with self._lock:
            self.emergency_active = False
            self.emergency_arm = None

    # ── Main Loop ─────────────────────────────────────────────────────────

    def start(self):
        self.running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        print("[Signal] Controller started.")
        return t

    def stop(self):
        self.running = False

    def _run(self):
        """Tick loop at 1Hz."""
        # Start first arm green
        self._begin_green(ARM_NAMES[0])

        while self.running:
            time.sleep(1.0)
            with self._lock:
                self._tick()

            if self.on_state_change:
                self.on_state_change(self.get_state())

    def _tick(self):
        """Advance state machine by one second."""
        if self.emergency_active and self.emergency_arm:
            self._handle_emergency()
            return

        if self.phase_timer is None or not self.phase_timer.expired:
            return  # still counting down

        # Phase expired — advance
        if self.in_yellow:
            self.in_yellow = False
            self.in_all_red = True
            self._set_all(SignalState.RED)
            self.phase_timer = PhaseTimer(ALL_RED_PHASE)

        elif self.in_all_red:
            self.in_all_red = False
            # Move to next arm
            self.current_arm_index = (self.current_arm_index + 1) % len(ARM_NAMES)
            self._begin_green(ARM_NAMES[self.current_arm_index])

        else:
            # Green expired → yellow
            arm = ARM_NAMES[self.current_arm_index]
            self.signals[arm] = SignalState.YELLOW
            self.in_yellow = True
            self.phase_timer = PhaseTimer(YELLOW_PHASE)
            # Check lane reversal after each green phase
            self._check_lane_reversal(arm)

    def _begin_green(self, arm: str):
        green_dur = self._compute_green_duration(arm)
        self.signals = {a: SignalState.RED for a in ARM_NAMES}
        self.signals[arm] = SignalState.GREEN
        self.in_yellow = False
        self.in_all_red = False
        self.phase_timer = PhaseTimer(green_dur)
        self.phase_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "arm": arm,
            "green_duration": green_dur,
            "density": dict(self.density),
        })

    def _compute_green_duration(self, arm: str) -> float:
        """
        Proportional allocation:
          arm_share = arm_density / total_density
          green = arm_share * (total_budget)
        Clamped to [MIN_GREEN, MAX_GREEN].
        Total budget = sum of arm green times in one full cycle cap.
        """
        total = sum(self.density.values())
        if total == 0:
            return MIN_GREEN

        share = self.density.get(arm, 0) / total
        # Budget: each arm gets up to MAX_GREEN proportionally
        raw = share * MAX_GREEN * len(ARM_NAMES)
        return round(max(MIN_GREEN, min(MAX_GREEN, raw)), 1)

    def _handle_emergency(self):
        """
        All arms red → emergency arm green for EMERGENCY_GREEN seconds.
        """
        arm = self.emergency_arm
        if arm is None:
            return

        # Set all red first
        self._set_all(SignalState.EMERGENCY_OVERRIDE)
        # Give emergency arm green
        self.signals[arm] = SignalState.GREEN

        if self.phase_timer is None:
            self.phase_timer = PhaseTimer(EMERGENCY_HOLD)

        if self.phase_timer.expired:
            self.emergency_active = False
            self.emergency_arm = None
            self.phase_timer = None
            self._log("Emergency cleared, resuming normal cycle")
            # Resume with next arm
            self._begin_green(ARM_NAMES[self.current_arm_index])

    # ── Lane Reversal Logic ───────────────────────────────────────────────

    def _check_lane_reversal(self, arm: str):
        """
        If one arm's density is 3× any other arm's density for 5 cycles, suggest reversal.
        """
        for other in ARM_NAMES:
            if other == arm:
                continue
            d_arm  = self.density.get(arm, 0)
            d_other = self.density.get(other, 0)
            if d_other > 0 and d_arm / d_other >= 3.0:
                self.cycle_counts[arm] = self.cycle_counts.get(arm, 0) + 1
                if self.cycle_counts[arm] >= 5:
                    self.cycle_counts[arm] = 0
                    suggestion = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "message": (
                            f"Lane reversal suggested: {arm} arm has 3× density of {other}. "
                            f"Consider reversing a lane from {other} to {arm}."
                        ),
                        "arm_high": arm,
                        "arm_low": other,
                        "density_high": d_arm,
                        "density_low": d_other,
                    }
                    self.lane_reversal_suggestions.append(suggestion)
                    self._log(suggestion["message"])
            else:
                self.cycle_counts[arm] = 0

    # ── Helpers ───────────────────────────────────────────────────────────

    def _set_all(self, state: SignalState):
        for a in ARM_NAMES:
            self.signals[a] = state

    def _log(self, msg: str):
        self._incidents.append({
            "timestamp": datetime.utcnow().isoformat(),
            "message": msg,
        })
        print(f"[Signal] {msg}")

    def get_state(self) -> dict:
        """Serialisable snapshot for WebSocket push."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "signals": {a: self.signals[a].value for a in ARM_NAMES},
            "active_arm": ARM_NAMES[self.current_arm_index],
            "countdown": round(self.phase_timer.remaining, 1) if self.phase_timer else 0,
            "emergency_active": self.emergency_active,
            "emergency_arm": self.emergency_arm,
            "density": dict(self.density),
            "lane_reversal_suggestions": self.lane_reversal_suggestions[-5:],
            "incidents": self._incidents[-20:],
            "phase": (
                "emergency" if self.emergency_active
                else "yellow" if self.in_yellow
                else "all_red" if self.in_all_red
                else "green"
            ),
        }


# ── CLI Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    def on_change(state):
        sigs = " | ".join(f"{a}:{state['signals'][a][0].upper()}" for a in ARM_NAMES)
        print(f"[{state['timestamp'][-8:-3]}] {sigs}  countdown={state['countdown']}s")

    ctrl = SignalController(on_state_change=on_change)

    # Simulate changing densities
    ctrl.update_density("North", 70)
    ctrl.update_density("South", 20)
    ctrl.update_density("East",  50)
    ctrl.update_density("West",  10)

    t = ctrl.start()
    try:
        time.sleep(10)
        print("\n>>> Simulating emergency on East arm <<<\n")
        ctrl.trigger_emergency("East")
        time.sleep(40)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.stop()
