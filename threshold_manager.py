"""
Adaptive threshold manager for anomaly detection.
Dynamically adjusts thresholds based on signal statistics.
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple
import logging
from dataclasses import dataclass
import time


logger = logging.getLogger(__name__)


@dataclass
class ThresholdStatistics:
    """Statistics for threshold calculation."""
    mean: float
    std: float
    percentile_95: float
    percentile_99: float
    min_val: float
    max_val: float
    samples_count: int
    last_update: float


class AdaptiveThresholdManager:
    """
    Manages adaptive thresholds for anomaly detection.
    Uses sliding window approach with exponential smoothing.
    """
    
    def __init__(self, window_size: int = 100, 
                 percentile: float = 95.0,
                 min_samples: int = 50,
                 update_rate: float = 0.1):
        """
        Initialize threshold manager.
        
        Args:
            window_size: Size of sliding window for statistics
            percentile: Percentile for threshold calculation
            min_samples: Minimum samples before threshold is valid
            update_rate: Exponential smoothing rate (0-1)
        """
        self.window_size = window_size
        self.percentile = percentile
        self.min_samples = min_samples
        self.update_rate = update_rate
        
        # Sliding window buffer
        self.score_buffer = deque(maxlen=window_size)
        
        # Current statistics
        self.stats = ThresholdStatistics(
            mean=0.0,
            std=0.0,
            percentile_95=0.0,
            percentile_99=0.0,
            min_val=float('inf'),
            max_val=float('-inf'),
            samples_count=0,
            last_update=time.time()
        )
        
        # Threshold history for analysis
        self.threshold_history = deque(maxlen=1000)
        
        # Current threshold
        self._current_threshold = None
        self._baseline_threshold = None
        
        logger.info(f"Threshold manager initialized with window_size={window_size}, "
                    f"percentile={percentile}")
    
    def update(self, anomaly_score: float) -> float:
        """
        Update threshold with new anomaly score.
        
        Args:
            anomaly_score: New anomaly score
            
        Returns:
            Updated threshold value
        """
        # Add to buffer
        self.score_buffer.append(anomaly_score)
        self.stats.samples_count += 1
        
        # Update statistics if we have enough samples
        if len(self.score_buffer) >= self.min_samples:
            self._update_statistics()
            self._update_threshold()
        
        return self.get_threshold()
    
    def _update_statistics(self):
        """Update internal statistics."""
        scores = np.array(self.score_buffer)
        
        # Calculate new statistics
        new_mean = np.mean(scores)
        new_std = np.std(scores)
        new_p95 = np.percentile(scores, 95)
        new_p99 = np.percentile(scores, 99)
        
        # Exponential smoothing
        alpha = self.update_rate
        self.stats.mean = alpha * new_mean + (1 - alpha) * self.stats.mean
        self.stats.std = alpha * new_std + (1 - alpha) * self.stats.std
        self.stats.percentile_95 = alpha * new_p95 + (1 - alpha) * self.stats.percentile_95
        self.stats.percentile_99 = alpha * new_p99 + (1 - alpha) * self.stats.percentile_99
        
        # Update min/max
        self.stats.min_val = min(self.stats.min_val, np.min(scores))
        self.stats.max_val = max(self.stats.max_val, np.max(scores))
        self.stats.last_update = time.time()
    
    def _update_threshold(self):
        """Update the anomaly threshold."""
        scores = np.array(self.score_buffer)
        
        # Calculate threshold based on percentile
        percentile_threshold = np.percentile(scores, self.percentile)
        
        # Alternative: mean + k*std approach
        std_threshold = self.stats.mean + 3 * self.stats.std
        
        # Use the more conservative threshold
        new_threshold = max(percentile_threshold, std_threshold)
        
        # Apply smoothing to threshold updates
        if self._current_threshold is None:
            self._current_threshold = new_threshold
            self._baseline_threshold = new_threshold
        else:
            self._current_threshold = (self.update_rate * new_threshold + 
                                     (1 - self.update_rate) * self._current_threshold)
        
        # Store in history
        self.threshold_history.append({
            'timestamp': time.time(),
            'threshold': self._current_threshold,
            'mean': self.stats.mean,
            'std': self.stats.std
        })
    
    def get_threshold(self) -> float:
        """
        Get current threshold value.
        
        Returns:
            Current threshold or default if not enough samples
        """
        if self._current_threshold is None:
            # Return a high default threshold until we have enough data
            return 10.0
        return self._current_threshold
    
    def is_anomaly(self, score: float, update: bool = True) -> bool:
        """
        Check if score indicates an anomaly.
        
        Args:
            score: Anomaly score to check
            update: Whether to update threshold with this score
            
        Returns:
            True if anomaly detected
        """
        if update and not self._is_outlier(score):
            # Only update with non-anomalous scores
            self.update(score)
        
        return score > self.get_threshold()
    
    def _is_outlier(self, score: float) -> bool:
        """
        Check if score is an outlier (for filtering updates).
        
        Args:
            score: Score to check
            
        Returns:
            True if outlier
        """
        if self.stats.samples_count < self.min_samples:
            return False
        
        # Consider as outlier if > 99th percentile
        return score > self.stats.percentile_99
    
    def get_statistics(self) -> Dict:
        """Get current statistics."""
        return {
            'threshold': self.get_threshold(),
            'mean': self.stats.mean,
            'std': self.stats.std,
            'percentile_95': self.stats.percentile_95,
            'percentile_99': self.stats.percentile_99,
            'min': self.stats.min_val,
            'max': self.stats.max_val,
            'samples_count': self.stats.samples_count,
            'buffer_size': len(self.score_buffer),
            'last_update': self.stats.last_update
        }
    
    def reset(self):
        """Reset threshold manager to initial state."""
        self.score_buffer.clear()
        self.threshold_history.clear()
        self._current_threshold = None
        self._baseline_threshold = None
        self.stats = ThresholdStatistics(
            mean=0.0,
            std=0.0,
            percentile_95=0.0,
            percentile_99=0.0,
            min_val=float('inf'),
            max_val=float('-inf'),
            samples_count=0,
            last_update=time.time()
        )
        logger.info("Threshold manager reset")
    
    def get_threshold_history(self, last_n: Optional[int] = None) -> list:
        """
        Get threshold history.
        
        Args:
            last_n: Number of recent entries to return
            
        Returns:
            List of threshold history entries
        """
        if last_n is None:
            return list(self.threshold_history)
        return list(self.threshold_history)[-last_n:]