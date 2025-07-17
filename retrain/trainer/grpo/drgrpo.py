"""
Dr. GRPO Implementation - GRPO Done Right

Dr. GRPO fixes two key biases in the original GRPO algorithm:
1. Response-level length bias: caused by dividing by response length |o_i|
2. Question-level difficulty bias: caused by dividing by std(rewards)

The fix is simple: remove these two normalization terms to get unbiased optimization.

Reference: "Understanding R1-Zero-Like Training: A Critical Perspective"
Section 3.2: "Dr. GRPO: Group Relative Policy Optimization Done Right"
"""

import logging
from typing import Dict, Any
import torch
import ray

# Import base GRPO implementation
from .grpo import BaseGRPO
from ...config_models import TrainingConfig

logger = logging.getLogger(__name__)


class BaseDRGRPO(BaseGRPO):
    """
    Base Dr. GRPO class - GRPO Done Right.
    
    Dr. GRPO removes two biases from the original GRPO algorithm:
    1. Length bias: No longer divides by response length |o_i|
    2. Standard deviation bias: No longer normalizes by std(rewards)
    
    This results in unbiased optimization that prevents:
    - Artificially favoring shorter correct responses
    - Artificially favoring longer incorrect responses  
    - Uneven weighting of different question difficulties
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize Dr. GRPO with the same configuration as base GRPO.
        
        The only difference is in the advantage computation which removes
        the problematic normalization terms.
        
        Args:
            config: Complete training configuration
        """
        super().__init__(config)
        logger.info("Dr. GRPO initialized - using unbiased advantage computation")
        
    def _group_relative_normalization(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Apply Dr. GRPO's unbiased group-relative normalization.
        
        Unlike standard GRPO, Dr. GRPO only subtracts the mean but does NOT
        divide by standard deviation. This removes the question-level difficulty bias.
        
        Original GRPO: (advantages - mean) / std  <-- biased
        Dr. GRPO:      (advantages - mean)        <-- unbiased
        
        Args:
            advantages: Raw advantage values
            
        Returns:
            Mean-centered advantages without std normalization
        """
        # Remove mean to center advantages (this is still beneficial)
        mean_adv = advantages.mean()
        
        # This removes the question-level difficulty bias
        return advantages - mean_adv
        
    def _compute_policy_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor, 
                           advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute policy loss using Dr. GRPO's unbiased approach.
        
        This method ensures that the loss computation does not include
        the response-level length bias that would come from dividing by |o_i|.
        
        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Previous policy log probabilities  
            advantages: Advantage values (already unbiased from Dr. GRPO normalization)
            
        Returns:
            Unbiased policy loss
        """
        # Compute probability ratios
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Dr. GRPO normalized advantages (without std division)
        normalized_advantages = self._group_relative_normalization(advantages)
        
        # Standard PPO clipped surrogate loss
        # Note: We do NOT divide by response length here (no 1/|o_i| bias)
        surr1 = ratio * normalized_advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * normalized_advantages
        
        return -torch.min(surr1, surr2).mean()
        
    async def health_check(self) -> Dict[str, Any]:
        """Health check with Dr. GRPO identification."""
        base_health = await super().health_check()
        
        # Add Dr. GRPO specific information
        dr_grpo_health = {
            'algorithm': 'dr_grpo',
            'bias_fixes': ['removed_length_normalization', 'removed_std_normalization'],
            'reference': 'Understanding R1-Zero-Like Training: A Critical Perspective'
        }
        
        return {**base_health, **dr_grpo_health}


# Ray Actor - Clean Dr. GRPO implementation 
@ray.remote(num_cpus=2, num_gpus=0)
class DRGRPO(BaseDRGRPO):
    """
    Dr. GRPO Ray Actor - extends BaseDRGRPO with Ray remote capabilities.
    
    GRPO Done Right: Simple fixes to remove length and std normalization biases.
    Results in better token efficiency and unbiased optimization.
    """
    pass  # Inherits all functionality from BaseDRGRPO


 