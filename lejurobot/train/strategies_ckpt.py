#!/usr/bin/env python

"""
Checkpoint saving strategies for XHUMAN training.

This module provides different strategies to determine when to save checkpoints
based on evaluation metrics. Each strategy can extract a specific loss from the
output_dict and compute a score to track the best checkpoint.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np

@dataclass
class EvalBatchStats:
    """
    Statistics computed from evaluation batch losses.
    
    This class stores per-batch losses and provides various statistical metrics
    that can be used by checkpoint strategies to determine model performance.
    
    Attributes:
        losses: List of loss values from each evaluation batch.
    """
    losses: list = field(default_factory=list)
    
    @property
    def mean(self) -> float:
        """Compute mean loss across all batches."""
        return float(np.mean(self.losses)) if self.losses else float('inf')
    
    @property
    def std(self) -> float:
        """Compute standard deviation of losses."""
        return float(np.std(self.losses)) if len(self.losses) > 1 else 0.0
    
    @property
    def min(self) -> float:
        """Compute minimum loss."""
        return float(np.min(self.losses)) if self.losses else float('inf')
    
    @property
    def max(self) -> float:
        """Compute maximum loss."""
        return float(np.max(self.losses)) if self.losses else float('inf')
    
    @property
    def percentile_90(self) -> float:
        """Compute 90th percentile loss."""
        return float(np.percentile(self.losses, 90)) if self.losses else float('inf')
    
    @property
    def percentile_95(self) -> float:
        """Compute 95th percentile loss."""
        return float(np.percentile(self.losses, 95)) if self.losses else float('inf')
    
    @property
    def median(self) -> float:
        """Compute median loss."""
        return float(np.median(self.losses)) if self.losses else float('inf')


class CheckpointStrategy(ABC):
    """
    Abstract base class for checkpoint saving strategies.
    
    A checkpoint strategy determines when to save a checkpoint based on
    evaluation metrics. It extracts a specific loss from the output_dict
    and computes a score that indicates model performance.
    """
    
    def __init__(self, name: str, loss_key: str = "loss", lower_is_better: bool = True):
        """
        Initialize checkpoint strategy.
        
        Args:
            name: Name identifier for this strategy.
            loss_key: Key to extract from output_dict (e.g., "loss", "mse_loss", "action_loss").
            lower_is_better: If True, lower scores are better; if False, higher scores are better.
        """
        self.name = name
        self.loss_key = loss_key
        self.lower_is_better = lower_is_better
        self.best_score = float('inf') if lower_is_better else float('-inf')
        self.best_step = -1
        self.history: list[tuple[int, float]] = []  # (step, score) history
    
    @abstractmethod
    def compute_score(self, batch_stats: EvalBatchStats) -> float:
        """
        Compute a score from evaluation batch statistics.
        
        Args:
            batch_stats: Statistics from evaluation batches.
            
        Returns:
            A scalar score that will be compared to determine best checkpoint.
        """
        pass
    
    def is_better(self, new_score: float) -> bool:
        """
        Determine if new_score is better than current best.
        
        Args:
            new_score: Score to compare against current best.
            
        Returns:
            True if new_score is better than best_score.
        """
        if self.lower_is_better:
            return new_score < self.best_score
        else:
            return new_score > self.best_score
    
    def update(self, batch_stats: EvalBatchStats, step: int) -> tuple[bool, float]:
        """
        Update best score if new score is better.
        
        Args:
            batch_stats: Statistics from evaluation batches.
            step: Current training step.
            
        Returns:
            Tuple of (is_new_best, score) where is_new_best indicates if checkpoint should be saved.
        """
        score = self.compute_score(batch_stats)
        self.history.append((step, score))
        
        if self.is_better(score):
            self.best_score = score
            self.best_step = step
            return True, score
        return False, score
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this strategy's current state."""
        return {
            "name": self.name,
            "loss_key": self.loss_key,
            "best_score": self.best_score,
            "best_step": self.best_step,
            "lower_is_better": self.lower_is_better,
        }


class MeanCheckpointStrategy(CheckpointStrategy):
    """
    Strategy 1: Save checkpoint based on mean loss.
    
    This is the simplest and most common strategy. It saves a checkpoint
    when the mean evaluation loss improves.
    """
    
    def __init__(self, loss_key: str = "loss"):
        super().__init__(name="best_mean", loss_key=loss_key, lower_is_better=True)
    
    def compute_score(self, batch_stats: EvalBatchStats) -> float:
        """Returns mean loss (lower is better)."""
        return batch_stats.mean


class StableCheckpointStrategy(CheckpointStrategy):
    """
    Strategy 2: Save checkpoint based on mean + λ*std.
    
    This strategy penalizes models with high variance in losses,
    preferring stable models even if they have slightly higher mean loss.
    
    Score = mean + lambda_penalty * std
    """
    
    def __init__(self, loss_key: str = "loss", lambda_penalty: float = 1.0):
        """
        Initialize stable checkpoint strategy.
        
        Args:
            loss_key: Key to extract from output_dict.
            lambda_penalty: Weight for standard deviation penalty (higher = more penalty for instability).
        """
        super().__init__(name="best_stable", loss_key=loss_key, lower_is_better=True)
        self.lambda_penalty = lambda_penalty
    
    def compute_score(self, batch_stats: EvalBatchStats) -> float:
        """Returns mean + lambda*std (lower is better)."""
        return batch_stats.mean + self.lambda_penalty * batch_stats.std
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["lambda_penalty"] = self.lambda_penalty
        return info


class RobustCheckpointStrategy(CheckpointStrategy):
    """
    Strategy 3: Save checkpoint based on 90th percentile loss.
    
    This strategy is more robust to outliers than mean, focusing on
    the upper tail of the loss distribution.
    """
    
    def __init__(self, loss_key: str = "loss", percentile: float = 90.0):
        """
        Initialize robust checkpoint strategy.
        
        Args:
            loss_key: Key to extract from output_dict.
            percentile: Percentile to use (default 90.0).
        """
        super().__init__(name="best_robust", loss_key=loss_key, lower_is_better=True)
        self.percentile = percentile
    
    def compute_score(self, batch_stats: EvalBatchStats) -> float:
        """Returns 90th percentile loss (lower is better)."""
        if self.percentile == 90.0:
            return batch_stats.percentile_90
        elif self.percentile == 95.0:
            return batch_stats.percentile_95
        elif self.percentile == 50.0:
            return batch_stats.median
        else:
            return float(np.percentile(batch_stats.losses, self.percentile)) if batch_stats.losses else float('inf')
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["percentile"] = self.percentile
        return info


class MinimaxCheckpointStrategy(CheckpointStrategy):
    """
    Strategy 4: Save checkpoint based on maximum loss (minimax).
    
    This strategy minimizes the worst-case loss, ensuring that
    even the hardest examples perform reasonably well.
    """
    
    def __init__(self, loss_key: str = "loss"):
        super().__init__(name="best_minimax", loss_key=loss_key, lower_is_better=True)
    
    def compute_score(self, batch_stats: EvalBatchStats) -> float:
        """Returns max loss (lower is better)."""
        return batch_stats.max


class MinCheckpointStrategy(CheckpointStrategy):
    """
    Strategy 5: Save checkpoint based on minimum loss.
    
    This strategy focuses on the best-case performance,
    tracking when the model performs exceptionally well on easy examples.
    """
    
    def __init__(self, loss_key: str = "loss"):
        super().__init__(name="best_min", loss_key=loss_key, lower_is_better=True)
    
    def compute_score(self, batch_stats: EvalBatchStats) -> float:
        """Returns min loss (lower is better)."""
        return batch_stats.min


class MedianCheckpointStrategy(CheckpointStrategy):
    """
    Strategy 6: Save checkpoint based on median loss.
    
    This strategy is robust to outliers like RobustCheckpointStrategy
    but focuses on the middle of the distribution.
    """
    
    def __init__(self, loss_key: str = "loss"):
        super().__init__(name="best_median", loss_key=loss_key, lower_is_better=True)
    
    def compute_score(self, batch_stats: EvalBatchStats) -> float:
        """Returns median loss (lower is better)."""
        return batch_stats.median


class CheckpointManager:
    """
    Manages multiple checkpoint strategies.
    
    This class coordinates multiple checkpoint strategies and handles
    saving checkpoints when any strategy indicates improvement.
    """
    
    def __init__(self, strategies: Optional[list[CheckpointStrategy]] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            strategies: List of checkpoint strategies to use. If None, uses default strategies.
        """
        if strategies is None:
            # Default strategies
            self.strategies = [
                MeanCheckpointStrategy(),
                StableCheckpointStrategy(lambda_penalty=1.0),
                RobustCheckpointStrategy(percentile=90.0),
                MinimaxCheckpointStrategy(),
            ]
        else:
            self.strategies = strategies
        
        # Create a dict for quick access by name
        self.strategies_dict = {s.name: s for s in self.strategies}
    
    def update_all(self, batch_stats: EvalBatchStats, step: int) -> Dict[str, tuple[bool, float]]:
        """
        Update all strategies and return which ones improved.
        
        Args:
            batch_stats: Statistics from evaluation batches.
            step: Current training step.
            
        Returns:
            Dictionary mapping strategy name to (is_new_best, score).
        """
        results = {}
        for strategy in self.strategies:
            is_new_best, score = strategy.update(batch_stats, step)
            results[strategy.name] = (is_new_best, score)
        return results
    
    def get_best_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about best scores for all strategies."""
        return {name: strategy.get_info() for name, strategy in self.strategies_dict.items()}
    
    def add_strategy(self, strategy: CheckpointStrategy):
        """Add a new strategy to the manager."""
        self.strategies.append(strategy)
        self.strategies_dict[strategy.name] = strategy
    
    def get_strategy(self, name: str) -> Optional[CheckpointStrategy]:
        """Get a specific strategy by name."""
        return self.strategies_dict.get(name)


def create_default_checkpoint_strategies(
    loss_key: str = "loss",
    lambda_penalty: float = 1.0,
    percentile: float = 90.0,
) -> list[CheckpointStrategy]:
    """
    Create the default set of checkpoint strategies.
    
    Args:
        loss_key: Key to extract from output_dict.
        lambda_penalty: Lambda for stable strategy.
        percentile: Percentile for robust strategy.
        
    Returns:
        List of checkpoint strategies.
    """
    return [
        MeanCheckpointStrategy(loss_key=loss_key),
        StableCheckpointStrategy(loss_key=loss_key, lambda_penalty=lambda_penalty),
        RobustCheckpointStrategy(loss_key=loss_key, percentile=percentile),
        MinimaxCheckpointStrategy(loss_key=loss_key),
    ]
