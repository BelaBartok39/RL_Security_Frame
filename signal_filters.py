"""
Signal preprocessing and filtering utilities.
Optimized for RF signal processing on edge devices.
"""

import numpy as np
import torch
from scipy import signal
from typing import Tuple, Dict, Optional, Union


class SignalPreprocessor:
    """
    Handles signal preprocessing including filtering, normalization,
    and feature extraction for RF signals.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create filters
        self.highpass_filter = self._create_highpass_filter()
        self.lowpass_filter = self._create_lowpass_filter()
        
        # Cache for filter states (for streaming)
        self.filter_state = {}
        
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'highpass_cutoff': 0.01,
            'lowpass_cutoff': 0.45,
            'filter_order': 4,
            'normalization': 'robust',  # 'robust', 'standard', 'minmax'
            'clip_sigma': 5.0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    def _create_highpass_filter(self) -> Dict:
        """Create highpass filter coefficients."""
        b, a = signal.butter(
            self.config['filter_order'], 
            self.config['highpass_cutoff'], 
            'high'
        )
        return {'b': b, 'a': a}
    
    def _create_lowpass_filter(self) -> Dict:
        """Create lowpass filter coefficients."""
        b, a = signal.butter(
            self.config['filter_order'], 
            self.config['lowpass_cutoff'], 
            'low'
        )
        return {'b': b, 'a': a}
    
    def remove_dc_offset(self, i_data: np.ndarray, q_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove DC offset using highpass filtering.
        
        Args:
            i_data: In-phase component
            q_data: Quadrature component
            
        Returns:
            Filtered I and Q data
        """
        i_filtered = signal.filtfilt(
            self.highpass_filter['b'], 
            self.highpass_filter['a'], 
            i_data
        )
        q_filtered = signal.filtfilt(
            self.highpass_filter['b'], 
            self.highpass_filter['a'], 
            q_data
        )
        return i_filtered, q_filtered
    
    def normalize_robust(self, data: np.ndarray) -> np.ndarray:
        """
        Robust normalization using median and MAD.
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data
        """
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        normalized = (data - median) / (1.4826 * mad + 1e-8)
        
        # Clip outliers
        clip_value = self.config['clip_sigma']
        return np.clip(normalized, -clip_value, clip_value)
    
    def normalize_standard(self, data: np.ndarray) -> np.ndarray:
        """
        Standard normalization using mean and std.
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data
        """
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        normalized = (data - mean) / std
        
        # Clip outliers
        clip_value = self.config['clip_sigma']
        return np.clip(normalized, -clip_value, clip_value)
    
    def normalize_minmax(self, data: np.ndarray) -> np.ndarray:
        """
        Min-max normalization to [0, 1].
        
        Args:
            data: Input data array
            
        Returns:
            Normalized data
        """
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val + 1e-8
        return (data - min_val) / range_val
    
    def preprocess_iq(self, i_data: np.ndarray, q_data: np.ndarray) -> torch.Tensor:
        """
        Complete preprocessing pipeline for I/Q data.
        
        Args:
            i_data: In-phase component
            q_data: Quadrature component
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Remove DC offset
        i_filtered, q_filtered = self.remove_dc_offset(i_data, q_data)
        
        # Normalize based on configuration
        norm_method = self.config['normalization']
        if norm_method == 'robust':
            i_norm = self.normalize_robust(i_filtered)
            q_norm = self.normalize_robust(q_filtered)
        elif norm_method == 'standard':
            i_norm = self.normalize_standard(i_filtered)
            q_norm = self.normalize_standard(q_filtered)
        elif norm_method == 'minmax':
            i_norm = self.normalize_minmax(i_filtered)
            q_norm = self.normalize_minmax(q_filtered)
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")
        
        # Convert to tensor with channels first (batch, channels, length)
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        # Stack I and Q as separate channels
        tensor_i = torch.from_numpy(i_norm)
        tensor_q = torch.from_numpy(q_norm)
        iq_tensor = torch.stack([tensor_i, tensor_q], dim=0)  # (2, length)
        iq_tensor = iq_tensor.unsqueeze(0).to(dtype=dtype, device=self.device)  # (1,2,length)
        return iq_tensor
    
    def extract_spectral_features(self, i_data: np.ndarray, q_data: np.ndarray) -> Dict[str, float]:
        """
        Extract spectral features from I/Q data.
        
        Args:
            i_data: In-phase component
            q_data: Quadrature component
            
        Returns:
            Dictionary of spectral features
        """
        # Complex signal
        complex_signal = i_data + 1j * q_data
        
        # Power spectral density
        freqs, psd = signal.periodogram(complex_signal, scaling='density')
        
        # Normalize PSD
        psd_norm = psd / (np.sum(psd) + 1e-10)
        
        # Calculate features step by step
        peak_frequency = freqs[np.argmax(psd)]
        spectral_centroid = np.sum(freqs * psd_norm)
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
        spectral_energy = np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        
        # Extract features
        features = {
            'peak_frequency': float(peak_frequency),
            'spectral_centroid': float(spectral_centroid),
            'spectral_bandwidth': float(spectral_bandwidth),
            'spectral_energy': float(spectral_energy),
            'spectral_entropy': float(spectral_entropy)
        }
        
        return features
    
    def extract_temporal_features(self, i_data: np.ndarray, q_data: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features from I/Q data.
        
        Args:
            i_data: In-phase component
            q_data: Quadrature component
            
        Returns:
            Dictionary of temporal features
        """
        # Magnitude
        magnitude = np.sqrt(i_data**2 + q_data**2)
        
        # Phase
        phase = np.angle(i_data + 1j * q_data)
        phase_diff = np.diff(np.unwrap(phase))
        
        features = {
            'magnitude_mean': np.mean(magnitude),
            'magnitude_std': np.std(magnitude),
            'magnitude_skew': self._skewness(magnitude),
            'magnitude_kurtosis': self._kurtosis(magnitude),
            'phase_mean': np.mean(phase),
            'phase_std': np.std(phase),
            'instantaneous_frequency_mean': np.mean(phase_diff),
            'instantaneous_frequency_std': np.std(phase_diff)
        }
        
        return features
    
    def extract_rf_puf_features(self, i_data: np.ndarray, q_data: np.ndarray) -> Dict[str, float]:
        """
        Extract features relevant for RF-PUF authentication.
        Based on the RF-PUF paper concepts.
        
        Args:
            i_data: In-phase component
            q_data: Quadrature component
            
        Returns:
            Dictionary of RF-PUF relevant features
        """
        # I-Q imbalance
        i_power = np.mean(i_data**2)
        q_power = np.mean(q_data**2)
        amplitude_imbalance = 10 * np.log10(i_power / (q_power + 1e-10))
        
        # Phase imbalance
        correlation = np.corrcoef(i_data, q_data)[0, 1]
        phase_imbalance = np.arccos(np.clip(correlation, -1, 1)) * 180 / np.pi - 90
        
        # DC offset (before filtering)
        i_dc = np.mean(i_data)
        q_dc = np.mean(q_data)
        
        # Frequency offset estimation (simplified)
        complex_signal = i_data + 1j * q_data
        phase_diff = np.angle(complex_signal[1:] * np.conj(complex_signal[:-1]))
        freq_offset = np.mean(phase_diff) / (2 * np.pi)
        
        features = {
            'amplitude_imbalance_db': amplitude_imbalance,
            'phase_imbalance_deg': phase_imbalance,
            'i_dc_offset': i_dc,
            'q_dc_offset': q_dc,
            'frequency_offset_normalized': freq_offset,
            'signal_power_dbm': 10 * np.log10(np.mean(i_data**2 + q_data**2) + 1e-10)
        }
        
        return features
    
    @staticmethod
    def _skewness(data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        return np.mean(((data - mean) / std) ** 4) - 3