"""
DARTSim RS-DRL Project - Core Implementation

This package contains the core implementation components:

- dartsim_env.py      : OfflineDARTSimEnv  — episode-replay env from pre-collected data
- live_dartsim_env.py : LiveDARTSimEnv     — live TCP env connecting to DARTSim Docker
- rs_drl_dqn.py       : RSDRLDQN          — DQN + randomised reward reshaping (RS-DRL)
- mape_k_architecture.py : MAPE-K architecture for adaptive systems

Workflow
--------
Online training (resolves action-coverage gap):
    1. Start container: .\\utils\\start_dartsim_live.ps1
    2. Train:           python scripts/train_online.py --timesteps 200000

Offline training (from pre-collected data):
    1. Collect data:    python scripts/collect_offline_data.py --use-tcp --episodes 5000
    2. Train:           python scripts/train_rs_drl.py --offline
"""

from src.dartsim_env import OfflineDARTSimEnv
from src.live_dartsim_env import LiveDARTSimEnv





