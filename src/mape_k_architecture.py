"""
MAPE-K Architecture for Self-Adaptive Systems
Implements Monitor-Analyze-Plan-Execute over a shared Knowledge base.

Based on RS-DRL paper's integration with MAPE-K for on-demand retraining.
Runtime verification (UPPAAL-SMC/ActivFORMS) is deferred to future work.

MAPE-K Components:
- Monitor: Collects runtime telemetry and quality metrics
- Analyzer: Detects deviations and threshold violations
- Planner: Selects adaptation actions/strategies
- Executor: Applies adaptations to the system
- Knowledge: Shared repository for models, metrics, thresholds
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path
import numpy as np
from collections import deque


class AdaptationTrigger(Enum):
    """Types of adaptation triggers."""
    THRESHOLD_VIOLATION = "threshold_violation"
    PERIODIC = "periodic"
    MANUAL = "manual"
    MODEL_DEGRADATION = "model_degradation"


@dataclass
class SystemMetrics:
    """Runtime system metrics collected by Monitor."""
    timestep: int
    reward: float
    mission_success: bool
    targets_detected: int
    team_destroyed: bool
    decision_time: float
    episode_length: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class Thresholds:
    """Quality thresholds for triggering adaptation."""
    min_reward: float = 0.7
    min_mission_success_rate: float = 0.8
    max_decision_time: float = 1000.0
    min_targets_detected_ratio: float = 0.6


@dataclass
class Knowledge:
    """
    Knowledge repository storing:
    - Trained RL models
    - Historical metrics
    - Current thresholds
    - Adaptation history
    """
    # Model storage
    current_model_path: Optional[str] = None
    model_history: List[Dict] = field(default_factory=list)
    
    # Metrics history
    metrics_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Thresholds
    thresholds: Thresholds = field(default_factory=Thresholds)
    
    # Adaptation history
    adaptations_performed: List[Dict] = field(default_factory=list)
    
    # Configuration
    retrain_threshold_violations: int = 3  # Number of violations before retraining
    retrain_interval: int = 1000  # Minimum timesteps between retrains


class Monitor:
    """
    Monitor component: Collects runtime telemetry and quality metrics.
    
    Continuously monitors the system and collects metrics for analysis.
    """
    
    def __init__(self, knowledge: Knowledge):
        """
        Initialize Monitor.
        
        Args:
            knowledge: Shared knowledge repository
        """
        self.knowledge = knowledge
        self.current_metrics = None
    
    def collect_metrics(
        self,
        timestep: int,
        reward: float,
        info: Dict[str, Any]
    ) -> SystemMetrics:
        """
        Collect metrics from environment step.
        
        Args:
            timestep: Current timestep
            reward: Reward received
            info: Info dictionary from environment
        
        Returns:
            SystemMetrics object
        """
        # Extract metrics from info
        results = info.get("results", {})
        state_data = info.get("state_data", {})
        
        metrics = SystemMetrics(
            timestep=timestep,
            reward=reward,
            mission_success=results.get("missionSuccess", False) if results else False,
            targets_detected=results.get("targetsDetected", 0) if results else 0,
            team_destroyed=results.get("destroyed", False) if results else False,
            decision_time=results.get("decisionTimeAvg", 0.0) if results else 0.0,
            episode_length=info.get("episode_step", 0)
        )
        
        self.current_metrics = metrics
        self.knowledge.metrics_history.append(metrics)
        
        return metrics
    
    def get_recent_metrics(self, window: int = 100) -> List[SystemMetrics]:
        """
        Get recent metrics within a window.
        
        Args:
            window: Number of recent metrics to return
        
        Returns:
            List of recent SystemMetrics
        """
        return list(self.knowledge.metrics_history)[-window:]
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current metrics."""
        return self.current_metrics


class Analyzer:
    """
    Analyzer component: Detects deviations and threshold violations.
    
    Analyzes collected metrics and determines if adaptation is needed.
    """
    
    def __init__(self, knowledge: Knowledge):
        """
        Initialize Analyzer.
        
        Args:
            knowledge: Shared knowledge repository
        """
        self.knowledge = knowledge
        self.violation_count = 0
        self.last_retrain_timestep = 0
    
    def analyze(self, metrics: SystemMetrics) -> Optional[AdaptationTrigger]:
        """
        Analyze metrics and determine if adaptation is needed.
        
        Args:
            metrics: Current system metrics
        
        Returns:
            AdaptationTrigger if adaptation needed, None otherwise
        """
        thresholds = self.knowledge.thresholds
        
        # Check threshold violations
        violations = []
        
        # Reward threshold
        if metrics.reward < thresholds.min_reward:
            violations.append("reward")
        
        # Mission success (computed over recent history)
        recent_metrics = self.knowledge.metrics_history
        if len(recent_metrics) >= 10:
            recent_successes = [m.mission_success for m in recent_metrics[-10:]]
            success_rate = np.mean(recent_successes)
            if success_rate < thresholds.min_mission_success_rate:
                violations.append("mission_success")
        
        # Decision time
        if metrics.decision_time > thresholds.max_decision_time:
            violations.append("decision_time")
        
        # Targets detected (if mission complete)
        if metrics.mission_success and metrics.targets_detected > 0:
            # Would need max_targets to compute ratio - simplified for now
            pass
        
        # Determine adaptation trigger
        if len(violations) > 0:
            self.violation_count += 1
            
            # Check if enough violations for retraining
            if (self.violation_count >= self.knowledge.retrain_threshold_violations and
                metrics.timestep - self.last_retrain_timestep >= self.knowledge.retrain_interval):
                
                return AdaptationTrigger.THRESHOLD_VIOLATION
        else:
            # Reset violation count if no violations
            self.violation_count = 0
        
        return None
    
    def should_retrain(self, trigger: AdaptationTrigger) -> bool:
        """
        Determine if retraining should be triggered.
        
        Args:
            trigger: Adaptation trigger
        
        Returns:
            True if retraining should be performed
        """
        if trigger == AdaptationTrigger.THRESHOLD_VIOLATION:
            return True
        return False
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary of current violations."""
        return {
            "violation_count": self.violation_count,
            "last_retrain": self.last_retrain_timestep
        }


class Planner:
    """
    Planner component: Selects adaptation actions/strategies.
    
    Plans adaptation actions based on analysis results.
    For RS-DRL, this includes:
    - Deciding when to retrain
    - Selecting hyperparameters for retraining
    - Choosing between different adaptation strategies
    """
    
    def __init__(self, knowledge: Knowledge):
        """
        Initialize Planner.
        
        Args:
            knowledge: Shared knowledge repository
        """
        self.knowledge = knowledge
    
    def plan_adaptation(
        self,
        trigger: AdaptationTrigger,
        current_metrics: SystemMetrics
    ) -> Dict[str, Any]:
        """
        Plan adaptation actions.
        
        Args:
            trigger: Adaptation trigger
            current_metrics: Current system metrics
        
        Returns:
            Adaptation plan dictionary
        """
        plan = {
            "trigger": trigger.value,
            "timestep": current_metrics.timestep,
            "timestamp": time.time()
        }
        
        if trigger == AdaptationTrigger.THRESHOLD_VIOLATION:
            # Plan retraining
            plan["action"] = "retrain"
            plan["retrain_config"] = {
                "total_timesteps": 5000,  # Shorter retraining
                "batch_size": 32,  # Batch size for offline RL training
                "rho": 0.3,  # Default RS-DRL rho
                "learning_rate": 1e-4,
                "continue_training": True  # Continue from current model
            }
        elif trigger == AdaptationTrigger.PERIODIC:
            # Periodic retraining (optional)
            plan["action"] = "retrain"
            plan["retrain_config"] = {
                "total_timesteps": 10000,
                "batch_size": 32,  # Batch size for offline RL training
                "rho": 0.3,
                "learning_rate": 1e-4,
                "continue_training": True
            }
        else:
            plan["action"] = "none"
        
        return plan
    
    def select_hyperparameters(self, metrics_history: List[SystemMetrics]) -> Dict[str, Any]:
        """
        Select hyperparameters based on performance history.
        
        Args:
            metrics_history: Historical metrics
        
        Returns:
            Hyperparameter configuration
        """
        # Simple strategy: use defaults for now
        # Could be enhanced with adaptive hyperparameter selection
        return {
            "rho": 0.3,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "batch_size": 32
        }


class Executor:
    """
    Executor component: Applies adaptations to the system.
    
    Executes adaptation plans, including:
    - Triggering model retraining
    - Updating system configuration
    - Applying runtime adaptations
    """
    
    def __init__(self, knowledge: Knowledge, model_manager=None):
        """
        Initialize Executor.
        
        Args:
            knowledge: Shared knowledge repository
            model_manager: Model manager for loading/saving models (optional)
        """
        self.knowledge = knowledge
        self.model_manager = model_manager
        self.last_retrain_timestep = 0
    
    def execute(
        self,
        plan: Dict[str, Any],
        current_model_path: Optional[str] = None,
        env=None
    ) -> Dict[str, Any]:
        """
        Execute adaptation plan.
        
        Args:
            plan: Adaptation plan from Planner
            current_model_path: Path to current model
            env: Environment for retraining (optional)
        
        Returns:
            Execution result dictionary
        """
        action = plan.get("action", "none")
        
        result = {
            "executed": False,
            "action": action,
            "timestep": plan.get("timestep"),
            "timestamp": time.time()
        }
        
        if action == "retrain":
            # Retrain model
            retrain_config = plan.get("retrain_config", {})
            
            if env is not None and current_model_path is not None:
                try:
                    # Load current model
                    import sys
                    from pathlib import Path
                    # Add src to path if needed
                    src_path = Path(__file__).parent.parent
                    if str(src_path) not in sys.path:
                        sys.path.insert(0, str(src_path))
                    
                    from src.rs_drl_dqn import RSDRLDQN
                    from stable_baselines3 import DQN
                    
                    try:
                        model = RSDRLDQN.load(current_model_path, env=env)
                    except:
                        model = DQN.load(current_model_path, env=env)
                    
                    # Continue training
                    # For offline RL: use model.train() instead of model.learn()
                    # Check if this is offline mode (OfflineDARTSimEnv)
                    is_offline = hasattr(env, 'transitions') or type(env).__name__ == 'OfflineDARTSimEnv'
                    
                    if is_offline:
                        # Offline RL: train directly from replay buffer
                        # Estimate gradient steps: assume buffer has enough data
                        total_timesteps = retrain_config.get("total_timesteps", 5000)
                        batch_size = retrain_config.get("batch_size", 32)
                        gradient_steps = max(1, total_timesteps // batch_size)
                        
                        # Train on existing replay buffer data
                        model.train(gradient_steps=gradient_steps, batch_size=batch_size)
                    else:
                        # Online RL: use model.learn() for environment interaction
                        model.learn(
                            total_timesteps=retrain_config.get("total_timesteps", 5000),
                            log_interval=100
                        )
                    
                    # Save updated model
                    new_model_path = current_model_path.replace(".zip", "_retrained.zip")
                    model.save(new_model_path)
                    
                    # Update knowledge
                    self.knowledge.current_model_path = new_model_path
                    self.knowledge.model_history.append({
                        "path": new_model_path,
                        "timestep": plan.get("timestep"),
                        "trigger": plan.get("trigger")
                    })
                    
                    self.last_retrain_timestep = plan.get("timestep", 0)
                    result["executed"] = True
                    result["new_model_path"] = new_model_path
                    
                except Exception as e:
                    result["error"] = str(e)
            else:
                result["error"] = "Environment or model path not provided"
        
        # Record adaptation
        self.knowledge.adaptations_performed.append(result)
        
        return result
    
    def get_last_retrain_timestep(self) -> int:
        """Get timestep of last retraining."""
        return self.last_retrain_timestep


class MAPEKManager:
    """
    MAPE-K Manager: Coordinates all MAPE-K components.
    
    Implements the MAPE-K control loop for self-adaptive systems.
    Integrates RS-DRL with on-demand retraining.
    """
    
    def __init__(
        self,
        knowledge: Knowledge,
        env=None,
        model_path: Optional[str] = None
    ):
        """
        Initialize MAPE-K Manager.
        
        Args:
            knowledge: Shared knowledge repository
            env: Environment for training (optional)
            model_path: Initial model path
        """
        self.knowledge = knowledge
        self.env = env
        self.model_path = model_path
        
        # Initialize components
        self.monitor = Monitor(knowledge)
        self.analyzer = Analyzer(knowledge)
        self.planner = Planner(knowledge)
        self.executor = Executor(knowledge)
        
        # Set initial model path in knowledge
        if model_path:
            self.knowledge.current_model_path = model_path
    
    def step(
        self,
        timestep: int,
        reward: float,
        info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute one MAPE-K cycle.
        
        Args:
            timestep: Current timestep
            reward: Reward received
            info: Info from environment
        
        Returns:
            Adaptation result if adaptation occurred, None otherwise
        """
        # Monitor: Collect metrics
        metrics = self.monitor.collect_metrics(timestep, reward, info)
        
        # Analyze: Detect violations
        trigger = self.analyzer.analyze(metrics)
        
        if trigger is not None:
            # Plan: Create adaptation plan
            plan = self.planner.plan_adaptation(trigger, metrics)
            
            # Execute: Apply adaptation
            result = self.executor.execute(
                plan,
                self.knowledge.current_model_path,
                self.env
            )
            
            return result
        
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current MAPE-K status."""
        return {
            "current_model": self.knowledge.current_model_path,
            "metrics_count": len(self.knowledge.metrics_history),
            "adaptations_count": len(self.knowledge.adaptations_performed),
            "violation_summary": self.analyzer.get_violation_summary(),
            "thresholds": {
                "min_reward": self.knowledge.thresholds.min_reward,
                "min_mission_success_rate": self.knowledge.thresholds.min_mission_success_rate,
                "max_decision_time": self.knowledge.thresholds.max_decision_time
            },
            "model_history_count": len(self.knowledge.model_history)
        }
    
    def save_knowledge(self, path: str):
        """Save knowledge repository to file."""
        # Convert deque to list for JSON serialization
        metrics_list = [
            {
                "timestep": m.timestep,
                "reward": m.reward,
                "mission_success": m.mission_success,
                "targets_detected": m.targets_detected,
                "team_destroyed": m.team_destroyed,
                "decision_time": m.decision_time,
                "episode_length": m.episode_length,
                "timestamp": m.timestamp
            }
            for m in self.knowledge.metrics_history
        ]
        
        knowledge_dict = {
            "current_model_path": self.knowledge.current_model_path,
            "model_history": self.knowledge.model_history,
            "metrics_history": metrics_list,
            "thresholds": {
                "min_reward": self.knowledge.thresholds.min_reward,
                "min_mission_success_rate": self.knowledge.thresholds.min_mission_success_rate,
                "max_decision_time": self.knowledge.thresholds.max_decision_time,
                "min_targets_detected_ratio": self.knowledge.thresholds.min_targets_detected_ratio
            },
            "adaptations_performed": self.knowledge.adaptations_performed,
            "retrain_threshold_violations": self.knowledge.retrain_threshold_violations,
            "retrain_interval": self.knowledge.retrain_interval
        }
        
        with open(path, 'w') as f:
            json.dump(knowledge_dict, f, indent=2)
    
    def load_knowledge(self, path: str):
        """Load knowledge repository from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.knowledge.current_model_path = data.get("current_model_path")
        self.knowledge.model_history = data.get("model_history", [])
        self.knowledge.adaptations_performed = data.get("adaptations_performed", [])
        
        # Reconstruct metrics
        for m_data in data.get("metrics_history", []):
            metrics = SystemMetrics(
                timestep=m_data["timestep"],
                reward=m_data["reward"],
                mission_success=m_data["mission_success"],
                targets_detected=m_data["targets_detected"],
                team_destroyed=m_data["team_destroyed"],
                decision_time=m_data["decision_time"],
                episode_length=m_data["episode_length"],
                timestamp=m_data.get("timestamp", time.time())
            )
            self.knowledge.metrics_history.append(metrics)
        
        # Load thresholds
        thresholds_data = data.get("thresholds", {})
        self.knowledge.thresholds = Thresholds(**thresholds_data)
        
        self.knowledge.retrain_threshold_violations = data.get("retrain_threshold_violations", 3)
        self.knowledge.retrain_interval = data.get("retrain_interval", 1000)

