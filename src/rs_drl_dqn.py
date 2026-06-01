"""
RS-DRL DQN Implementation
Extends Stable-Baselines3 DQN with Randomized Reward Reshaping (RS-DRL).

Implements Algorithm 2 from the RS-DRL paper for offline RL training:
- Randomly select up to ρ fraction of failed transitions (reward <= 0)
- Replace their reward with optimistic value (default: 1.0)
- Apply during minibatch replay processing

Designed for offline RL training from pre-collected data (plan.md Phase 3).
The reward reshaping is applied during gradient steps when sampling from
the replay buffer populated with offline collected transitions.

Reference: RS-DRL paper Algorithm 2 (RewardShaping function)
"""

import numpy as np
import os

# Reduce PyTorch memory usage during import (helps with paging file issues)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch
from typing import Any, Dict, List, Optional, Tuple
from stable_baselines3.dqn import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import polyak_update


class RSDRLRewardShaping:
    """
    Reward reshaping function (Algorithm 2 from RS-DRL paper).
    
    Randomly selects up to ρ fraction of failed transitions (reward == 0)
    and replaces their reward with optimistic value 1.
    """
    
    def __init__(self, rho: float = 0.3, optimistic_reward: float = 1.0):
        """
        Initialize reward reshaping.
        
        Args:
            rho: Reshaping factor (fraction of failed transitions to reshape)
            optimistic_reward: Optimistic reward value (default: 1.0)
        """
        self.rho = rho
        self.optimistic_reward = optimistic_reward
    
    def reshape_rewards(self, rewards: np.ndarray) -> np.ndarray:
        """
        Apply reward reshaping to a reward array (Algorithm 2).
        
        Args:
            rewards: Array of rewards
        
        Returns:
            Reshaped rewards array
        """
        rewards = rewards.copy()
        N = len(rewards)
        
        # Find failed transitions (reward == 0 or <= 0)
        # Using <= 0.0 threshold for failed transitions
        failed_indices = np.where(rewards <= 0.0)[0]
        
        if len(failed_indices) == 0:
            return rewards
        
        # Compute K = ⌊ρN⌋
        K = int(self.rho * N)
        K = min(K, len(failed_indices))
        
        if K == 0:
            return rewards
        
        # Randomly select K failed transitions
        np.random.shuffle(failed_indices)
        selected_indices = failed_indices[:K]
        
        # Replace with optimistic reward
        rewards[selected_indices] = self.optimistic_reward
        
        return rewards


class RSDRLDQN(DQN):
    """
    RS-DRL DQN: Extends Stable-Baselines3 DQN with randomized reward reshaping.
    
    This class overrides the training step to apply reward reshaping
    during minibatch processing, as described in Algorithm 2 of the RS-DRL paper.
    """
    
    def __init__(
        self,
        *args,
        rho: float = 0.3,
        optimistic_reward: float = 1.0,
        **kwargs
    ):
        """
        Initialize RS-DRL DQN.
        
        Args:
            *args: Arguments passed to DQN.__init__
            rho: Reshaping factor (fraction of failed transitions to reshape)
            optimistic_reward: Optimistic reward value for reshaped transitions
            **kwargs: Keyword arguments passed to DQN.__init__
        """
        super().__init__(*args, **kwargs)
        
        # Initialize reward reshaping
        self.reward_shaping = RSDRLRewardShaping(
            rho=rho,
            optimistic_reward=optimistic_reward
        )
        self.rho = rho
        self.optimistic_reward = optimistic_reward
        
        # Statistics tracking
        self.num_reshaped_transitions = 0
        self.total_transitions = 0
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        Train the agent with reward reshaping.
        
        Overrides parent train method to apply reward reshaping to rewards
        before computing Q-targets.
        
        Args:
            gradient_steps: Number of gradient steps to perform
            batch_size: Batch size
        """
        # Reset statistics periodically
        # Use _n_updates if available, otherwise track manually
        if not hasattr(self, '_n_updates'):
            self._n_updates = 0
        if self._n_updates % 1000 == 0:
            self.num_reshaped_transitions = 0
            self.total_transitions = 0
        
        # Set training mode
        self.policy.set_training_mode(True)
        
        # Update learning rate if logger is available
        # For offline training, logger might not be initialized, so check first
        try:
            if hasattr(self, '_logger') and self._logger is not None:
                self._update_learning_rate(self.policy.optimizer)
            elif hasattr(self, 'logger') and self.logger is not None:
                # Try alternative logger attribute name
                self._update_learning_rate(self.policy.optimizer)
        except (AttributeError, RuntimeError):
            # Logger not initialized - skip learning rate update logging
            # The actual learning rate is still managed by the optimizer
            pass
        
        losses = []
        import torch as th
        
        for step in range(gradient_steps):
            # Sample from replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Apply reward reshaping (Algorithm 2)
            # Convert rewards to numpy for processing
            rewards_np = replay_data.rewards.cpu().numpy() if hasattr(replay_data.rewards, 'cpu') else np.array(replay_data.rewards)
            original_rewards = rewards_np.copy()
            reshaped_rewards = self.reward_shaping.reshape_rewards(rewards_np)
            
            # Track statistics
            self.total_transitions += len(original_rewards)
            num_reshaped = np.sum(reshaped_rewards != original_rewards)
            self.num_reshaped_transitions += num_reshaped
            
            # Update rewards in replay_data
            import torch as th
            if hasattr(replay_data.rewards, 'device'):
                reshaped_rewards_tensor = th.tensor(reshaped_rewards, dtype=th.float32, device=replay_data.rewards.device)
                # ReplayBufferSamples might be a NamedTuple - create new instance
                if hasattr(type(replay_data), '_fields'):
                    # NamedTuple - create new instance with updated rewards
                    fields = type(replay_data)._fields
                    kwargs = {field: getattr(replay_data, field) for field in fields}
                    kwargs['rewards'] = reshaped_rewards_tensor
                    replay_data = type(replay_data)(**kwargs)
                elif hasattr(replay_data, '__dict__'):
                    # Regular class - try dataclass replace
                    from dataclasses import replace
                    replay_data = replace(replay_data, rewards=reshaped_rewards_tensor)
                else:
                    # Fallback: modify in place if possible
                    object.__setattr__(replay_data, 'rewards', reshaped_rewards_tensor)
            
            # Continue with standard DQN training (from parent)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma
            
            with th.no_grad():
                # Compute next Q-values using target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy
                next_q_values, _ = th.max(next_q_values, dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # Compute target Q value
                if isinstance(discounts, th.Tensor):
                    discounts_tensor = discounts.reshape(-1, 1)
                else:
                    discounts_tensor = th.tensor(discounts, device=next_q_values.device)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts_tensor * next_q_values
            
            # Get current Q values
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            
            # Compute loss
            loss = th.nn.functional.mse_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            
            # Optimize
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            # Update target network
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            
            self._n_updates += 1
        
        # Log losses if logger available
        if hasattr(self, 'logger') and self.logger is not None:
            self.logger.record("train/loss", np.mean(losses))
            self.logger.record("train/reshaping_rate", 
                             self.num_reshaped_transitions / max(self.total_transitions, 1))
    
    def get_reshaping_stats(self) -> Dict[str, float]:
        """
        Get statistics about reward reshaping.
        
        Returns:
            Dictionary with reshaping statistics
        """
        if self.total_transitions == 0:
            return {
                "reshaping_rate": 0.0,
                "rho": self.rho
            }
        
        return {
            "reshaping_rate": self.num_reshaped_transitions / self.total_transitions,
            "rho": self.rho,
            "total_transitions": self.total_transitions,
            "reshaped_transitions": self.num_reshaped_transitions
        }


class RSDRLTrainingCallback(BaseCallback):
    """
    Callback to log RS-DRL specific metrics during training.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.reshaping_stats_history = []
    
    def _on_step(self) -> bool:
        """Log reshaping statistics."""
        if isinstance(self.model, RSDRLDQN):
            stats = self.model.get_reshaping_stats()
            self.reshaping_stats_history.append(stats)
            
            # Log to tensorboard if available
            if self.verbose > 0 and len(self.reshaping_stats_history) % 100 == 0:
                if hasattr(self.logger, 'record'):
                    self.logger.record("train/reshaping_rate", stats["reshaping_rate"])
        
        return True

