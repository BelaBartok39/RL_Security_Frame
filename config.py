"""
System configuration module.
Centralized configuration for all components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os


@dataclass
class ReceiverConfig:
    """UDP receiver configuration."""
    port: int = 12345
    buffer_size: int = 65536
    bind_address: str = '0.0.0.0'
    queue_size: int = 1000


@dataclass
class PreprocessingConfig:
    """Signal preprocessing configuration."""
    window_size: int = 1024
    overlap: float = 0.5
    sample_rate: float = 2e6  # 2 MHz default
    highpass_cutoff: float = 0.01
    lowpass_cutoff: float = 0.45
    filter_order: int = 4
    normalization: str = 'robust'
    clip_sigma: float = 5.0


@dataclass
class AnomalyDetectionConfig:
    """Anomaly detection configuration."""
    model_path: Optional[str] = None
    latent_dim: int = 32
    beta: float = 1.0
    device: str = 'cuda'
    # Adaptive threshold parameters
    threshold_window_size: int = 100
    threshold_percentile: float = 95.0
    threshold_min_samples: int = 50
    threshold_update_rate: float = 0.1


@dataclass
class ClassifierConfig:
    """Jammer classifier configuration."""
    model_path: Optional[str] = None
    jammer_types: List[str] = field(default_factory=lambda: [
        'barrage', 'tone', 'sweep', 'pulse', 'protocol_aware'
    ])
    confidence_threshold: float = 0.7


@dataclass
class ChannelScannerConfig:
    """Channel scanner configuration."""
    scan_channels: List[int] = field(default_factory=lambda: [
        2412000000,  # 2.412 GHz (WiFi Ch 1)
        2437000000,  # 2.437 GHz (WiFi Ch 6)
        2462000000,  # 2.462 GHz (WiFi Ch 11)
        5180000000,  # 5.180 GHz (5GHz band)
        5220000000,  # 5.220 GHz
        5745000000,  # 5.745 GHz
    ])
    scan_duration: float = 0.1  # seconds per channel
    energy_threshold_db: float = -70.0
    tx_port: int = 12346  # Port for sending channel change commands


@dataclass
class SystemConfig:
    """Main system configuration."""
    receiver: ReceiverConfig = field(default_factory=ReceiverConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    anomaly_detection: AnomalyDetectionConfig = field(default_factory=AnomalyDetectionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    channel_scanner: ChannelScannerConfig = field(default_factory=ChannelScannerConfig)
    
    # System-wide settings
    log_level: str = 'INFO'
    save_anomalies: bool = True
    anomaly_save_path: str = './anomalies'
    metrics_update_interval: float = 1.0
    
    @classmethod
    def from_file(cls, filepath: str) -> 'SystemConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        if 'receiver' in data:
            config.receiver = ReceiverConfig(**data['receiver'])
        if 'preprocessing' in data:
            config.preprocessing = PreprocessingConfig(**data['preprocessing'])
        if 'anomaly_detection' in data:
            config.anomaly_detection = AnomalyDetectionConfig(**data['anomaly_detection'])
        if 'classifier' in data:
            config.classifier = ClassifierConfig(**data['classifier'])
        if 'channel_scanner' in data:
            config.channel_scanner = ChannelScannerConfig(**data['channel_scanner'])
        
        # System-wide settings
        for key in ['log_level', 'save_anomalies', 'anomaly_save_path', 'metrics_update_interval']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'receiver': self.receiver.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'anomaly_detection': self.anomaly_detection.__dict__,
            'classifier': self.classifier.__dict__,
            'channel_scanner': self.channel_scanner.__dict__,
            'log_level': self.log_level,
            'save_anomalies': self.save_anomalies,
            'anomaly_save_path': self.anomaly_save_path,
            'metrics_update_interval': self.metrics_update_interval
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)