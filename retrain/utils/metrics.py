"""
Lean metrics tracking for retrain training loops.

This module provides a simple MetricsTracker that collects environment-specific
metrics during training, designed to integrate with TRL's existing logging.
"""

from typing import Dict, List, Any, Optional, Union
from collections import defaultdict, deque
import time
from dataclasses import dataclass, field
from loguru import logger

@dataclass
class GenerationMetrics:
    """Metrics for a single LLM generation/rollout."""
    
    # Generation quality
    parsing_success: bool
    tool_calls_attempted: int
    tool_calls_successful: int
    final_answer_provided: bool
    
    # Reward breakdown
    total_reward: float
    reward_per_step: List[float] = field(default_factory=list)
    
    # Generation details
    num_turns: int = 0
    completion_reason: str = "unknown"  # "terminated", "truncated", "max_steps"
    tools_used: List[str] = field(default_factory=list)  # Tools used during generation
    
    # Timing
    generation_time_seconds: Optional[float] = None

@dataclass 
class TrainingBatchMetrics:
    """Aggregated metrics for a training batch/iteration."""
    
    # Success rates
    parsing_success_rate: float
    tool_success_rate: float
    final_answer_rate: float
    completion_rate: float  # terminated vs truncated
    
    # Reward statistics
    mean_total_reward: float
    std_total_reward: float
    mean_reward_per_turn: float
    
    # Efficiency
    mean_turns_per_rollout: float
    mean_generation_time: float
    
    # Tool usage
    mean_tool_calls_per_rollout: float
    tool_usage_distribution: Dict[str, int] = field(default_factory=dict)

class MetricsTracker:
    """
    Lean metrics tracker for environment-specific training metrics.
    
    Designed to work alongside TRL's built-in metrics logging.
    Tracks generation quality, reward linkage, and training progress.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of recent generations to keep for moving averages
        """
        self.window_size = window_size
        
        # Recent generations for moving averages
        self._recent_generations: deque = deque(maxlen=window_size)
        
        # Cumulative counters
        self._total_generations = 0
        self._total_parsing_errors = 0
        self._total_tool_calls = 0
        self._total_successful_tool_calls = 0
        self._total_final_answers = 0
        
        # Per-iteration batch metrics
        self._batch_metrics_history: List[TrainingBatchMetrics] = []
        
        # Tool usage tracking
        self._tool_usage_counts: Dict[str, int] = defaultdict(int)
        
        # Training timing
        self._training_start_time: Optional[float] = None
        self._last_log_time: Optional[float] = None

    def start_training(self) -> None:
        """Mark the start of training for timing metrics."""
        self._training_start_time = time.time()
        self._last_log_time = time.time()
        logger.info("[MetricsTracker] Started tracking training metrics")

    def track_generation(self, generation_metrics: GenerationMetrics) -> None:
        """Track metrics from a single generation/rollout."""
        self._recent_generations.append(generation_metrics)
        self._total_generations += 1
        
        # Update cumulative counters
        if not generation_metrics.parsing_success:
            self._total_parsing_errors += 1
            
        self._total_tool_calls += generation_metrics.tool_calls_attempted
        self._total_successful_tool_calls += generation_metrics.tool_calls_successful
        
        if generation_metrics.final_answer_provided:
            self._total_final_answers += 1

    def track_tool_usage(self, tool_name: str) -> None:
        """Track usage of a specific tool."""
        self._tool_usage_counts[tool_name] += 1

    def compute_batch_metrics(self, generations: List[GenerationMetrics]) -> TrainingBatchMetrics:
        """Compute aggregated metrics for a batch of generations."""
        if not generations:
            return TrainingBatchMetrics(
                parsing_success_rate=0.0,
                tool_success_rate=0.0, 
                final_answer_rate=0.0,
                completion_rate=0.0,
                mean_total_reward=0.0,
                std_total_reward=0.0,
                mean_reward_per_turn=0.0,
                mean_turns_per_rollout=0.0,
                mean_generation_time=0.0,
                mean_tool_calls_per_rollout=0.0
            )
        
        n = len(generations)
        
        # Success rates
        parsing_success_rate = sum(g.parsing_success for g in generations) / n
        
        total_tools = sum(g.tool_calls_attempted for g in generations)
        successful_tools = sum(g.tool_calls_successful for g in generations)
        tool_success_rate = successful_tools / total_tools if total_tools > 0 else 0.0
        
        final_answer_rate = sum(g.final_answer_provided for g in generations) / n
        completion_rate = sum(g.completion_reason == "terminated" for g in generations) / n
        
        # Reward statistics
        rewards = [g.total_reward for g in generations]
        mean_total_reward = sum(rewards) / n
        std_total_reward = (sum((r - mean_total_reward) ** 2 for r in rewards) / n) ** 0.5
        
        # Calculate mean reward per turn
        all_step_rewards = []
        for g in generations:
            all_step_rewards.extend(g.reward_per_step)
        mean_reward_per_turn = sum(all_step_rewards) / len(all_step_rewards) if all_step_rewards else 0.0
        
        # Efficiency metrics
        mean_turns_per_rollout = sum(g.num_turns for g in generations) / n
        
        generation_times = [g.generation_time_seconds for g in generations if g.generation_time_seconds is not None]
        mean_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0.0
        
        mean_tool_calls_per_rollout = sum(g.tool_calls_attempted for g in generations) / n
        
        # Tool usage distribution (for this batch)
        tool_distribution = defaultdict(int)
        for g in generations:
            if hasattr(g, 'tools_used'):  # Will add this field when tracking
                for tool in g.tools_used:
                    tool_distribution[tool] += 1
        
        return TrainingBatchMetrics(
            parsing_success_rate=parsing_success_rate,
            tool_success_rate=tool_success_rate,
            final_answer_rate=final_answer_rate,
            completion_rate=completion_rate,
            mean_total_reward=mean_total_reward,
            std_total_reward=std_total_reward,
            mean_reward_per_turn=mean_reward_per_turn,
            mean_turns_per_rollout=mean_turns_per_rollout,
            mean_generation_time=mean_generation_time,
            mean_tool_calls_per_rollout=mean_tool_calls_per_rollout,
            tool_usage_distribution=dict(tool_distribution)
        )

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics for logging."""
        if not self._recent_generations:
            return {}
        
        # Compute metrics from recent window
        recent_batch = self.compute_batch_metrics(list(self._recent_generations))
        
        # Training progress metrics
        current_time = time.time()
        training_time = current_time - self._training_start_time if self._training_start_time else 0
        
        metrics = {
            # Environment-specific metrics with "env/" prefix
            "env/parsing_success_rate": recent_batch.parsing_success_rate,
            "env/tool_success_rate": recent_batch.tool_success_rate,
            "env/final_answer_rate": recent_batch.final_answer_rate,
            "env/completion_rate": recent_batch.completion_rate,
            
            # Reward linking metrics
            "env/mean_total_reward": recent_batch.mean_total_reward,
            "env/std_total_reward": recent_batch.std_total_reward,
            "env/mean_reward_per_turn": recent_batch.mean_reward_per_turn,
            
            # Efficiency metrics
            "env/mean_turns_per_rollout": recent_batch.mean_turns_per_rollout,
            "env/mean_generation_time": recent_batch.mean_generation_time,
            "env/mean_tool_calls_per_rollout": recent_batch.mean_tool_calls_per_rollout,
            
            # Cumulative progress
            "env/total_generations": self._total_generations,
            "env/total_parsing_errors": self._total_parsing_errors,
            "env/cumulative_parsing_error_rate": self._total_parsing_errors / self._total_generations if self._total_generations > 0 else 0,
            "env/cumulative_tool_success_rate": self._total_successful_tool_calls / self._total_tool_calls if self._total_tool_calls > 0 else 0,
            
            # Training timing
            "env/training_time_minutes": training_time / 60,
        }
        
        # Add top tool usage
        if self._tool_usage_counts:
            sorted_tools = sorted(self._tool_usage_counts.items(), key=lambda x: x[1], reverse=True)
            for i, (tool_name, count) in enumerate(sorted_tools[:3]):  # Top 3 tools
                metrics[f"env/tool_usage_{tool_name}"] = count
        
        return metrics

    def log_batch_summary(self, iteration: int, batch_metrics: TrainingBatchMetrics) -> None:
        """Log a summary of the current batch metrics."""
        self._batch_metrics_history.append(batch_metrics)
        
        logger.info(
            f"[MetricsTracker] Iteration {iteration} Summary:\n"
            f"  • Parsing Success: {batch_metrics.parsing_success_rate:.1%}\n"
            f"  • Tool Success: {batch_metrics.tool_success_rate:.1%}\n" 
            f"  • Final Answer Rate: {batch_metrics.final_answer_rate:.1%}\n"
            f"  • Mean Reward: {batch_metrics.mean_total_reward:.3f} ± {batch_metrics.std_total_reward:.3f}\n"
            f"  • Avg Turns/Rollout: {batch_metrics.mean_turns_per_rollout:.1f}\n"
            f"  • Generation Time: {batch_metrics.mean_generation_time:.2f}s"
        )

    def get_training_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of training progress over all iterations."""
        if not self._batch_metrics_history:
            return {}
        
        # Compute trends from batch history
        recent_batches = self._batch_metrics_history[-10:]  # Last 10 batches
        early_batches = self._batch_metrics_history[:10] if len(self._batch_metrics_history) >= 20 else self._batch_metrics_history[:len(self._batch_metrics_history)//2]
        
        def safe_mean(values):
            return sum(values) / len(values) if values else 0
        
        # Calculate trends
        recent_reward = safe_mean([b.mean_total_reward for b in recent_batches])
        early_reward = safe_mean([b.mean_total_reward for b in early_batches])
        reward_improvement = recent_reward - early_reward
        
        recent_success = safe_mean([b.parsing_success_rate for b in recent_batches])
        early_success = safe_mean([b.parsing_success_rate for b in early_batches])
        success_improvement = recent_success - early_success
        
        return {
            "training/total_iterations": len(self._batch_metrics_history),
            "training/reward_improvement": reward_improvement,
            "training/success_rate_improvement": success_improvement,
            "training/recent_mean_reward": recent_reward,
            "training/recent_success_rate": recent_success,
        } 