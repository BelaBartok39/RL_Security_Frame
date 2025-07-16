"""
Improved RF Autoencoder for anomaly detection.
Optimized for Jetson deployment with mixed precision support.
Based on VAE architecture from "Jamming Detection in MIMO-OFDM ISAC Systems Using Variational Autoencoders"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class ImprovedRFAutoencoder(nn.Module):
    """
    Variational Autoencoder with residual connections and attention mechanisms.
    
    This model is designed to learn normal RF signal patterns and detect
    anomalies through reconstruction error analysis.
    """
    
    def __init__(self, input_size: int = 1024, latent_dim: int = 32, beta: float = 1.0):
        """
        Initialize the VAE.
        
        Args:
            input_size: Size of input signal window
            latent_dim: Dimension of latent representation
            beta: Weight for KL divergence term (beta-VAE)
        """
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder layers
        self.encoder = self._build_encoder()
        
        # Attention mechanism
        self.attention = self._build_attention()
        
        # Variational layers
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.flatten = nn.Flatten()
        
        # Separate layers for mean and log variance
        self.mu_fc = nn.Linear(128 * 8, latent_dim)
        self.logvar_fc = nn.Linear(128 * 8, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8))
        )
        
        self.decoder = self._build_decoder()
        
    def _build_encoder(self) -> nn.ModuleDict:
        """Build encoder layers with residual connections."""
        return nn.ModuleDict({
            'conv1': nn.Sequential(
                nn.Conv1d(2, 32, 7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1)
            ),
            'conv2': nn.Sequential(
                nn.Conv1d(32, 64, 5, stride=2, padding=2),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1)
            ),
            'conv3': nn.Sequential(
                nn.Conv1d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1)
            )
        })
    
    def _build_attention(self) -> nn.Sequential:
        """Build attention mechanism for feature weighting."""
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()
        )
    
    def _build_decoder(self) -> nn.ModuleDict:
        """Build decoder layers."""
        return nn.ModuleDict({
            'conv1': nn.Sequential(
                nn.ConvTranspose1d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1)
            ),
            'conv2': nn.Sequential(
                nn.ConvTranspose1d(64, 32, 5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(0.1)
            ),
            'conv3': nn.Sequential(
                nn.ConvTranspose1d(32, 2, 7, stride=2, padding=3, output_padding=1),
                nn.Upsample(size=self.input_size, mode='linear', align_corners=False)
            )
        })
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch, 2, input_size)
            
        Returns:
            Tuple of (mu, logvar)
        """
        # Encoder with residual connections
        e1 = self.encoder['conv1'](x)
        e2 = self.encoder['conv2'](e1)
        e3 = self.encoder['conv3'](e2)
        
        # Apply attention
        att_weights = self.attention(e3).unsqueeze(2)
        e3_attended = e3 * att_weights
        
        # Pool and flatten
        pooled = self.pool(e3_attended)
        flat = self.flatten(pooled)
        
        # Get distribution parameters
        mu = self.mu_fc(flat)
        logvar = self.logvar_fc(flat)
        
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector
            
        Returns:
            Reconstructed signal
        """
        # Decoder
        d = self.decoder_fc(z)
        d1 = self.decoder['conv1'](d)
        d2 = self.decoder['conv2'](d1)
        reconstruction = self.decoder['conv3'](d2)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor of shape (batch, 2, input_size)
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return reconstruction, mu, logvar
    
    def loss_function(self, x: torch.Tensor, reconstruction: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate VAE loss with reconstruction and KL divergence terms.
        
        Args:
            x: Original input
            reconstruction: Reconstructed input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    @torch.jit.export
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate anomaly score based on reconstruction probability.
        
        Args:
            x: Input tensor
            
        Returns:
            Anomaly score
        """
        with torch.no_grad():
            reconstruction, mu, logvar = self.forward(x)
            
            # Calculate reconstruction error
            recon_error = torch.mean((x - reconstruction) ** 2, dim=(1, 2))
            
            # Calculate KL divergence
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            
            # Combined anomaly score
            anomaly_score = recon_error + 0.1 * kl_div
            
        return anomaly_score