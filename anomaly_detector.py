"""
Anomaly detection module using VAE and adaptive thresholds.
Integrates the autoencoder with threshold management.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass
import time
import os

from autoencoder import ImprovedRFAutoencoder
from threshold_manager import AdaptiveThresholdManager
from config import AnomalyDetectionConfig


logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    reconstruction_error: float
    kl_divergence: float
    timestamp: float
    confidence: float  # 0-1, how confident we are this is an anomaly


class AnomalyDetector:
    """
    Main anomaly detection module combining VAE and adaptive thresholds.
    """
    
    def __init__(self, config: AnomalyDetectionConfig):
        """
        Initialize anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize VAE
        self.model = ImprovedRFAutoencoder(
            input_size=1024,  # This should match preprocessing window size
            latent_dim=config.latent_dim,
            beta=config.beta
        ).to(self.device)
        
        # Load pre-trained model if available
        if config.model_path and os.path.exists(config.model_path):
            self.load_model(config.model_path)
        else:
            logger.warning("No pre-trained model found, using untrained VAE")
        
        # Initialize threshold manager
        self.threshold_manager = AdaptiveThresholdManager(
            window_size=config.threshold_window_size,
            percentile=config.threshold_percentile,
            min_samples=config.threshold_min_samples,
            update_rate=config.threshold_update_rate
        )
        
        # Metrics tracking
        self.total_samples = 0
        self.anomaly_count = 0
        self.last_anomaly_time = None
        self.anomaly_history = []
        
        # Set model to evaluation mode
        self.model.eval()
        
        logger.info(f"Anomaly detector initialized on {self.device}")
    
    def detect(self, preprocessed_tensor: torch.Tensor) -> AnomalyResult:
        """
        Detect anomalies in preprocessed RF data.
        
        Args:
            preprocessed_tensor: Preprocessed I/Q tensor from SignalPreprocessor
            
        Returns:
            AnomalyResult with detection details
        """
        timestamp = time.time()
        self.total_samples += 1
        
        with torch.no_grad():
            # Move to device if needed
            if preprocessed_tensor.device != self.device:
                preprocessed_tensor = preprocessed_tensor.to(self.device)
            
            # Get anomaly score from model
            anomaly_score = self.model.get_anomaly_score(preprocessed_tensor)
            
            # Get detailed reconstruction for analysis
            reconstruction, mu, logvar = self.model(preprocessed_tensor)
            
            # Calculate components
            recon_error = torch.mean((preprocessed_tensor - reconstruction) ** 2).item()
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).item()
            
            # Convert anomaly score to scalar
            score = anomaly_score.item()
            
            # Check against threshold
            is_anomaly = self.threshold_manager.is_anomaly(score)
            threshold = self.threshold_manager.get_threshold()
            
            # Calculate confidence (how far above threshold)
            if is_anomaly:
                confidence = min(1.0, (score - threshold) / threshold)
            else:
                confidence = 0.0
            
            # Create result
            result = AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=score,
                threshold=threshold,
                reconstruction_error=recon_error,
                kl_divergence=kl_div,
                timestamp=timestamp,
                confidence=confidence
            )
            
            # Update tracking
            if is_anomaly:
                self.anomaly_count += 1
                self.last_anomaly_time = timestamp
                self._record_anomaly(result)
            
            return result
    
    def _record_anomaly(self, result: AnomalyResult):
        """Record anomaly in history."""
        self.anomaly_history.append({
            'timestamp': result.timestamp,
            'score': result.anomaly_score,
            'threshold': result.threshold,
            'confidence': result.confidence
        })
        
        # Keep only recent history (last 1000 anomalies)
        if len(self.anomaly_history) > 1000:
            self.anomaly_history.pop(0)
    
    def get_metrics(self) -> Dict:
        """Get detection metrics."""
        anomaly_rate = self.anomaly_count / max(1, self.total_samples)
        
        metrics = {
            'total_samples': self.total_samples,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': anomaly_rate,
            'last_anomaly_time': self.last_anomaly_time,
            'threshold_stats': self.threshold_manager.get_statistics(),
            'model_device': str(self.device),
            'recent_anomalies': len(self.anomaly_history)
        }
        
        return metrics
    
    def train_on_normal_data(self, normal_data_loader: torch.utils.data.DataLoader, 
                           epochs: int = 10,
                           learning_rate: float = 1e-3):
        """
        Train the VAE on normal data.
        
        Args:
            normal_data_loader: DataLoader with normal RF samples
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for batch in normal_data_loader:
                if batch.device != self.device:
                    batch = batch.to(self.device)
                
                # Forward pass
                reconstruction, mu, logvar = self.model(batch)
                
                # Calculate loss
                loss_dict = self.model.loss_function(batch, reconstruction, mu, logvar)
                loss = loss_dict['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.model.eval()
        logger.info("Training completed")
    
    def save_model(self, filepath: str):
        """Save model state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold_stats': self.threshold_manager.get_statistics(),
            'metrics': self.get_metrics()
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
    
    def reset_threshold(self):
        """Reset adaptive threshold (useful after environment changes)."""
        self.threshold_manager.reset()
        logger.info("Threshold reset")
    
    def get_anomaly_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get anomaly history."""
        if last_n is None:
            return self.anomaly_history
        return self.anomaly_history[-last_n:]