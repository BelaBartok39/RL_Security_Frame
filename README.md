# RF Anomaly Detection System

A modularized framework for real-time RF signal anomaly detection, jamming classification, and adaptive channel hopping. Designed for deployment on NVIDIA Jetson platforms with HackRF SDR.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   HackRF RX     │────▶│  Signal Buffer   │────▶│ Preprocessing  │
│  (UDP Stream)   │     │   & Windowing    │     │  & Features     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                ┌──────────────────────────▼────────┐
                                │      Anomaly Detection Layer      │
                                │  ┌──────────────┐ ┌─────────────┐ │
                                │  │ Autoencoder  │ │  Threshold  │ │
                                │  │     (VAE)    │ │   Manager   │ │
                                │  └──────┬───────┘ └─────────────┘ │
                                └─────────┼─────────────────────────┘
                                         │ If Anomaly Detected
                                ┌────────▼─────────┐
                                │  DRN Classifier  │
                                │  (Jammer Type)   │
                                └────────┬─────────┘
                                         │
                                ┌────────▼─────────┐     ┌──────────────┐
                                │ Channel Scanner  │────▶│ HackRF TX    │
                                │ (Find Clean Ch.) │     │ (Change Ch.) │
                                └──────────────────┘     └──────────────┘
```

## Features

- **Real-time RF Signal Processing**: Receives I/Q data via UDP from HackRF
- **Advanced Anomaly Detection**: Variational Autoencoder (VAE) with attention mechanisms
- **Adaptive Thresholding**: Dynamic threshold adjustment based on signal statistics
- **Jammer Classification**: Deep Residual Network (DRN) for identifying jamming types
- **Intelligent Channel Hopping**: Automatic channel switching when jamming detected
- **Comprehensive GUI**: Real-time monitoring and control interface
- **Modular Design**: Easily configurable and extensible components

## System Requirements

- **Hardware**:
  - NVIDIA Jetson Orin Nano Super (or compatible)
  - HackRF One SDR
  - Minimum 8GB RAM
  
- **Software**:
  - Python 3.8+
  - CUDA 11.4+ (for GPU acceleration)
  - PyTorch 1.12+
  - NumPy, SciPy, Matplotlib
  - tkinter (for GUI)

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd rf-anomaly-detection
```

2. Install dependencies:
```bash
pip install torch torchvision numpy scipy matplotlib
```

3. Create default configuration:
```bash
python main.py --create-config
```

## Quick Start

### 1. First Time Setup (with Training)

Make the quick start script executable:
```bash
chmod +x quickstart.sh
```

Run the system for the first time (will automatically train VAE if needed):
```bash
./quickstart.sh
```

The system will:
1. Check GPU availability and optimize for Jetson
2. Collect normal RF data for 2 minutes if no training data exists
3. Train the VAE model on normal data (uses GPU acceleration)
4. Start the monitoring GUI

### 2. Normal Operation

After initial training, just run:
```bash
./quickstart.sh
```

Or use the startup script directly:
```bash
python3 startup.py
```

### 3. Test Mode (No Hardware Required)

Test the system with simulated RF data:
```bash
./quickstart.sh --test
```

### 4. Headless Mode (No GUI)

For production deployment without GUI:
```bash
./quickstart.sh --no-gui
```

### 5. Training Only Mode

To collect data and train models without running the system:
```bash
./quickstart.sh --train-only
```

## Configuration

Edit `config.json` to customize:

- **Receiver Settings**: UDP port, buffer size
- **Preprocessing**: Window size, overlap, filtering parameters
- **Anomaly Detection**: Model parameters, adaptive threshold settings
- **Classifier**: Jammer types, confidence thresholds
- **Channel Scanner**: Available channels, scan parameters

Example configuration snippet:
```json
{
  "receiver": {
    "port": 12345,
    "buffer_size": 65536
  },
  "preprocessing": {
    "window_size": 1024,
    "overlap": 0.5,
    "normalization": "robust"
  },
  "anomaly_detection": {
    "latent_dim": 32,
    "threshold_percentile": 95.0,
    "device": "cuda"
  }
}
```

## Module Descriptions

### Core Modules

- **`receiver.py`**: UDP packet receiver with threading and buffering
- **`protocols.py`**: Network protocol definitions for RF data transmission
- **`signal_filters.py`**: Signal preprocessing and feature extraction
- **`autoencoder.py`**: VAE implementation for anomaly detection
- **`threshold_manager.py`**: Adaptive threshold management
- **`anomaly_detector.py`**: Main anomaly detection logic
- **`jammer_classifier.py`**: DRN-based jammer type classification
- **`channel_scanner.py`**: Channel scanning and hopping logic
- **`processing_pipeline.py`**: Main processing pipeline orchestration
- **`gui.py`**: Real-time monitoring GUI
- **`config.py`**: Configuration management
- **`main.py`**: Application entry point

### Data Flow

1. **Reception**: HackRF streams I/Q samples via UDP
2. **Buffering**: Samples are buffered and windowed with configurable overlap
3. **Preprocessing**: DC offset removal, filtering, normalization
4. **Feature Extraction**: FFT-based spectral features, temporal features
5. **Anomaly Detection**: VAE reconstruction error compared to adaptive threshold
6. **Classification**: If anomaly detected, classify jammer type
7. **Mitigation**: Find clean channel and command HackRF to hop

## GUI Features

The monitoring GUI provides:

- **Real-time Plot**: Anomaly scores and adaptive threshold
- **Anomaly List**: Detected anomalies with classification results
- **Channel Status**: Energy levels and jamming status for all channels
- **System Log**: Comprehensive event logging
- **Controls**: Start/stop pipeline, reset thresholds, trigger channel sweeps

## Extending the System

### Adding New Jammer Types

1. Update `config.py` with new jammer types
2. Modify `jammer_classifier.py` to handle new classifications
3. Add training data for the new jammer types

### Custom Feature Extraction

Add new feature extraction methods to `signal_filters.py`:
```python
def extract_custom_features(self, i_data: np.ndarray, q_data: np.ndarray) -> Dict[str, float]:
    # Your custom feature extraction logic
    pass
```

### Alternative Anomaly Detection Models

Replace the VAE in `autoencoder.py` with your own model while maintaining the interface:
```python
class CustomAnomalyModel(nn.Module):
    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        # Your anomaly scoring logic
        pass
```

## Training Models

To train the VAE on normal signal data:

```python
from processing_pipeline import RFProcessingPipeline
from torch.utils.data import DataLoader

# Load your normal signal dataset
normal_data_loader = DataLoader(normal_dataset, batch_size=32)

# Train the model
pipeline.anomaly_detector.train_on_normal_data(normal_data_loader, epochs=50)

# Save trained models
pipeline.save_models("./trained_models")
```

## Troubleshooting

### No Data Received
- Check HackRF is transmitting to correct UDP port
- Verify network connectivity
- Check firewall settings

### High False Positive Rate
- Reset adaptive threshold after environment changes
- Increase threshold percentile in configuration
- Collect more normal data for training

### GPU Memory Issues
- Reduce window size in configuration
- Use CPU mode by setting `"device": "cpu"`
- Enable mixed precision training

## GPU Optimization for Jetson

The system is optimized for NVIDIA Jetson platforms:

1. **Automatic GPU Detection**: The system automatically detects and uses CUDA if available
2. **Mixed Precision Training**: Uses FP16 for faster training on Jetson
3. **Performance Mode**: The quickstart script sets Jetson to maximum performance
4. **Memory Management**: Periodic GPU cache clearing to prevent OOM errors

To verify GPU usage:
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
nvidia-smi  # or tegrastats on Jetson
```

## Training the Models

### VAE Training (Anomaly Detection)

The VAE needs to be trained on normal (non-jammed) RF signals:

```bash
# Automatic training on first run
python3 startup.py

# Or manual training
python3 training.py --collect --train-vae --epochs 50
```

The VAE learns the characteristics of normal signals and detects anomalies based on reconstruction error.

### DRN Classifier Training

To train the jammer classifier, you need labeled data for different jammer types:

```bash
# Prepare labeled data in data/ directory:
# - data/barrage/*.npy
# - data/tone/*.npy  
# - data/sweep/*.npy
# - data/pulse/*.npy

# Train classifier
python3 training.py --train-drn --data-path data/ --epochs 30
```

## License

[Your License Here]

## Acknowledgments

- Based on concepts from "Jamming Detection in MIMO-OFDM ISAC Systems Using Variational Autoencoders"
- Inspired by RF-PUF authentication techniques
- Designed for real-world deployment on edge AI platforms