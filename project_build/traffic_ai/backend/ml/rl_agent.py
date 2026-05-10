"""
Step 3: Reinforcement Learning Agent
Custom OpenAI Gym environment for traffic signal optimization.
Reward = negative total vehicle waiting time.
Uses Stable-Baselines3 PPO.
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

ARM_NAMES = ["North", "South", "East", "West"]
N_ARMS = 4
MIN_GREEN = 15
MAX_GREEN = 90


# ── Custom Gym Environment ────────────────────────────────────────────────────

class TrafficJunctionEnv(gym.Env):
    """
    Observation: density_score per arm (4 floats, 0-100)
                 + current_active_arm (one-hot, 4 floats)
                 + time_in_phase (1 float, 0-1 normalised)
                 = 9 floats total

    Action: Discrete(5)
        0 = extend current green by 5s
        1 = keep (no change)
        2 = shorten green by 5s
        3 = switch to next arm now
        4 = emergency hold (all red)

    Reward: - (sum of density scores × time_step)
            i.e. lower density = higher reward
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 500):
        super().__init__()
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # Internal state
        self._density = np.zeros(N_ARMS, dtype=np.float32)
        self._active_arm = 0
        self._phase_duration = float(MIN_GREEN)
        self._phase_elapsed = 0.0
        self._step_count = 0
        self._total_wait = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random initial densities
        self._density = self.np_random.uniform(0, 80, size=N_ARMS).astype(np.float32)
        self._active_arm = 0
        self._phase_duration = float(MIN_GREEN)
        self._phase_elapsed = 0.0
        self._step_count = 0
        self._total_wait = 0.0
        return self._obs(), {}

    def step(self, action: int):
        self._step_count += 1
        self._phase_elapsed += 1.0

        # Apply action
        if action == 0:   # extend
            self._phase_duration = min(MAX_GREEN, self._phase_duration + 5)
        elif action == 2: # shorten
            self._phase_duration = max(MIN_GREEN, self._phase_duration - 5)
        elif action == 3: # switch now
            self._phase_elapsed = self._phase_duration  # force expiry
        elif action == 4: # emergency hold
            self._phase_duration = max(MIN_GREEN, self._phase_duration - 10)
        # action == 1: no change

        # Simulate traffic evolution (simplified queuing model)
        for i in range(N_ARMS):
            if i == self._active_arm:
                # Active arm: vehicles leaving (discharge rate ~ 5/tick)
                self._density[i] = max(0.0, self._density[i] - self.np_random.uniform(3, 7))
            else:
                # Other arms: vehicles arriving (arrival rate ~ 2/tick)
                self._density[i] = min(100.0, self._density[i] + self.np_random.uniform(0, 4))

        # Phase expiry → advance arm
        if self._phase_elapsed >= self._phase_duration:
            self._active_arm = (self._active_arm + 1) % N_ARMS
            self._phase_duration = self._compute_adaptive_green()
            self._phase_elapsed = 0.0

        # Reward: negative total waiting (sum of non-active arm densities)
        waiting = sum(self._density[i] for i in range(N_ARMS) if i != self._active_arm)
        reward = -waiting / (100.0 * (N_ARMS - 1))   # normalised [-1, 0]
        self._total_wait += waiting

        terminated = self._step_count >= self.max_steps
        info = {
            "total_wait": self._total_wait,
            "density": self._density.tolist(),
            "active_arm": ARM_NAMES[self._active_arm],
        }
        return self._obs(), float(reward), terminated, False, info

    def _obs(self) -> np.ndarray:
        density_norm = self._density / 100.0
        one_hot = np.eye(N_ARMS, dtype=np.float32)[self._active_arm]
        elapsed_norm = np.array(
            [min(1.0, self._phase_elapsed / max(1, self._phase_duration))],
            dtype=np.float32
        )
        return np.concatenate([density_norm, one_hot, elapsed_norm])

    def _compute_adaptive_green(self) -> float:
        total = self._density.sum()
        if total == 0:
            return float(MIN_GREEN)
        share = self._density[self._active_arm] / total
        return float(np.clip(share * MAX_GREEN * N_ARMS, MIN_GREEN, MAX_GREEN))


# ── Training Script ────────────────────────────────────────────────────────────

def train(total_timesteps: int = 100_000, save_path: str = "models/rl_traffic_agent"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    env = TrafficJunctionEnv(max_steps=500)
    check_env(env, warn=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=os.path.dirname(save_path),
        name_prefix="rl_checkpoint",
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
    )

    print(f"[RL] Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)
    model.save(save_path)
    print(f"[RL] Model saved → {save_path}.zip")
    return model


def load_agent(path: str = "models/rl_traffic_agent") -> PPO:
    """Load a pretrained agent."""
    return PPO.load(path)


def infer_action(model: PPO, density: dict) -> dict:
    """
    Given density dict {"North":50,"South":20,...} and current active arm,
    return recommended action.
    """
    env = TrafficJunctionEnv()
    obs, _ = env.reset()
    # Inject real densities
    for i, arm in enumerate(ARM_NAMES):
        env._density[i] = density.get(arm, 0.0)
    obs = env._obs()
    action, _ = model.predict(obs, deterministic=True)
    action_map = {0:"extend", 1:"keep", 2:"shorten", 3:"switch", 4:"hold"}
    return {
        "action": int(action),
        "action_name": action_map[int(action)],
        "obs": obs.tolist(),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        train(total_timesteps=50_000)

    elif mode == "test":
        model = load_agent("models/rl_traffic_agent")
        result = infer_action(
            model,
            {"North": 80, "South": 10, "East": 40, "West": 5}
        )
        print(json.dumps(result, indent=2))
