"""
DARTSim Gymnasium Environment Adapter
Implements a Gymnasium-compatible RL environment for DARTSim.

For offline RL training - no TCP connection or DARTSim library required.
Uses pre-collected offline data to simulate the environment.

Episode-replay design
---------------------
Transitions are loaded as complete episode sequences (split on done=True
boundaries) and paired with their corresponding mission results. On reset()
a random episode is chosen; on step() the environment advances one recorded
transition regardless of the agent's chosen action. This preserves episode
structure and ensures mission outcomes (missionSuccess, targetsDetected, …)
are faithfully reported.

The agent's action is still accepted (and stored in info) so the API is
compatible with Stable-Baselines3, but the next state/reward/done come
from the recorded trajectory rather than a lookup table.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import json
import random


class OfflineDARTSimEnv(gym.Env):
    """
    Episode-replay offline RL environment backed by pre-collected DARTSim data.
    """

    metadata = {"render_modes": [], "render_fps": 4}

    # Action mapping (8 discrete actions for the DQN head)
    ACTIONS = [
        "IncAlt",   # 0
        "DecAlt",   # 1
        "IncAlt2",  # 2
        "DecAlt2",  # 3
        "GoTight",  # 4
        "GoLoose",  # 5
        "EcmOn",    # 6
        "EcmOff",   # 7
    ]

    ACTION_MAP = {
        "IncAlt": 0, "DecAlt": 1, "IncAlt2": 2, "DecAlt2": 3,
        "GoTight": 4, "GoLoose": 5, "EcmOn": 6, "EcmOff": 7,
        # Aliases present in collected data
        "Unknown": 0, "Finished": 0,
    }

    def __init__(
        self,
        obs_dim: int = 17,
        data_dir: Optional[str] = None,
        scenario: Optional[str] = None,
        max_transitions: Optional[int] = None,  # kept for API compat; bounds episodes loaded
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.data_dir = data_dir
        self.scenario = scenario
        self.max_transitions = max_transitions
        self.seed = seed

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)

        # Episode storage: list of dicts {"transitions": [...], "results": {...}}
        self.episodes: List[Dict] = []

        # Runtime state
        self._current_episode: List[Dict] = []
        self._current_results: Dict = {}
        self._step_idx: int = 0
        self.current_state: Optional[np.ndarray] = None

        # Legacy: flat transitions list kept so training code that reads
        # env.transitions still works (e.g. offline RL buffer population).
        self.transitions: List[Dict] = []
        # Removed transition_lookup — no longer used.

        self.sensor_noise = 0.0

        if data_dir is not None:
            self._load_episodes(data_dir, scenario, max_transitions)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_episodes(
        self,
        data_dir: str,
        scenario: Optional[str],
        max_transitions: Optional[int],
    ) -> None:
        """Load episodes as complete sequences paired with mission results."""
        data_path = Path(data_dir)
        pattern = f"episodes_{scenario}_*.json" if scenario else "episodes_*.json"
        episode_files = sorted(data_path.glob(pattern))

        if not episode_files:
            raise FileNotFoundError(f"No episode files found matching {pattern} in {data_dir}")

        print(f"Loading offline data: {len(episode_files)} files...")

        total_transitions = 0
        skipped_dim = 0

        for ep_file in episode_files:
            # Derive the matching results file (same stem prefix, different type)
            results_file = data_path / ep_file.name.replace("episodes_", "results_")
            try:
                raw_transitions = json.load(open(ep_file))
            except Exception as e:
                print(f"Warning: could not load {ep_file.name}: {e}")
                continue

            # Load per-episode results if available
            episode_results: List[Dict] = []
            if results_file.exists():
                try:
                    episode_results = json.load(open(results_file))
                except Exception:
                    episode_results = []

            # Split flat transition list into episodes on done=True boundaries
            current_ep: List[Dict] = []
            ep_index = 0

            for raw in raw_transitions:
                # Dimension guard
                if len(raw["state"]) != self.obs_dim or len(raw["next_state"]) != self.obs_dim:
                    skipped_dim += 1
                    # If mid-episode we drop the whole partial episode
                    if current_ep:
                        current_ep = []
                    continue

                transition = {
                    "state": np.array(raw["state"], dtype=np.float32),
                    "action": self.ACTION_MAP.get(raw.get("action", "Unknown"), 0),
                    "action_str": raw.get("action", "Unknown"),
                    "reward": float(raw["reward"]),
                    "next_state": np.array(raw["next_state"], dtype=np.float32),
                    "done": bool(raw["done"]),
                    "info": raw.get("info", {}),
                }
                current_ep.append(transition)
                self.transitions.append(transition)  # legacy flat list
                total_transitions += 1

                if raw["done"]:
                    results = (
                        episode_results[ep_index]
                        if ep_index < len(episode_results)
                        else {}
                    )
                    self.episodes.append({
                        "transitions": current_ep,
                        "results": results,
                    })
                    current_ep = []
                    ep_index += 1

                if max_transitions and total_transitions >= max_transitions:
                    break

            # Any incomplete episode at end of file is discarded
            if max_transitions and total_transitions >= max_transitions:
                break

        if skipped_dim:
            print(f"Skipped {skipped_dim} transitions with wrong observation dimension.")
        print(
            f"Loaded {len(self.episodes)} complete episodes "
            f"({total_transitions} transitions) from offline data."
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if not self.episodes:
            self.current_state = np.zeros(self.observation_space.shape, dtype=np.float32)
            return self.current_state.copy(), {}

        ep = random.choice(self.episodes)
        self._current_episode = ep["transitions"]
        self._current_results = ep["results"]
        self._step_idx = 0

        self.current_state = self._current_episode[0]["state"].copy()
        return self.current_state.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance one step along the current recorded episode.

        The *agent's* chosen action is accepted for API compatibility but the
        next state, reward and done flag are taken from the recorded trajectory.
        This is the only valid evaluation mode when no live simulator is available.
        """
        if not self._current_episode:
            # Not reset yet — auto-reset
            self.reset()

        if self._step_idx >= len(self._current_episode):
            # Episode already finished — return terminal step
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {"results": self._current_results}

        transition = self._current_episode[self._step_idx]
        self._step_idx += 1

        next_obs = transition["next_state"].copy()
        
        # Apply sensor noise failure injection (zeroing out threat sensors 7-11)
        if self.sensor_noise > 0.0:
            for idx in range(7, 12):
                if random.random() < self.sensor_noise:
                    next_obs[idx] = 0.0

        self.current_state = next_obs

        terminated = transition["done"]
        truncated = False
        info: Dict = {
            "recorded_action": transition["action_str"],
            "agent_action": int(action),
        }
        if terminated:
            info["results"] = self._current_results

        return self.current_state.copy(), transition["reward"], terminated, truncated, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        self.episodes.clear()
        self.transitions.clear()
        self.current_state = None
