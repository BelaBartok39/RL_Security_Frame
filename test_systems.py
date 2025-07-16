"""
Test script for RF anomaly detection system.
Generates synthetic RF data to test the pipeline without HackRF hardware.
"""

import socket
import struct
import time
import numpy as np
import threading
import logging
from typing import Tuple

from protocols import RFProtocol


logger = logging.getLogger(__name__)


class RFDataSimulator:
    """
    Simulates RF data for testing the system.
    Generates normal signals and various types of jamming.
    """
    
    def __init__(self, target_host: str = 'localhost', target_port: int = 12345):
        """
        Initialize simulator.
        
        Args:
            target_host: Host to send UDP packets to
            target_port: Port to send UDP packets to
        """
        self.target_host = target_host
        self.target_port = target_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.running = False
        self.current_mode = 'normal'
        self.sample_rate = 2e6  # 2 MHz
        self.packet_size = 1024  # samples per packet
        
        logger.info(f"RF simulator initialized, target: {target_host}:{target_port}")
    
    def generate_normal_signal(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate normal QPSK-like signal."""
        # Random QPSK symbols
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=num_samples//4)
        
        # Upsample
        upsampled = np.repeat(symbols, 4)
        
        # Add noise
        noise_power = 0.1
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(upsampled)) + 
                                         1j * np.random.randn(len(upsampled)))
        
        signal = upsampled[:num_samples] + noise[:num_samples]
        
        return np.real(signal), np.imag(signal)
    
    def generate_tone_jammer(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate tone jamming signal."""
        t = np.arange(num_samples) / self.sample_rate
        
        # Single tone at offset frequency
        freq_offset = 100e3  # 100 kHz
        jammer = 2.0 * np.exp(1j * 2 * np.pi * freq_offset * t)
        
        # Add small noise
        noise = 0.05 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        signal = jammer + noise
        return np.real(signal), np.imag(signal)
    
    def generate_sweep_jammer(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate frequency sweep jamming signal."""
        t = np.arange(num_samples) / self.sample_rate
        
        # Sweep from -500 kHz to +500 kHz
        sweep_rate = 1e6  # 1 MHz/s
        instantaneous_freq = sweep_rate * t - 500e3
        phase = 2 * np.pi * np.cumsum(instantaneous_freq) / self.sample_rate
        
        jammer = 1.5 * np.exp(1j * phase)
        
        # Add noise
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        signal = jammer + noise
        return np.real(signal), np.imag(signal)
    
    def generate_pulse_jammer(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pulse jamming signal."""
        signal = np.zeros(num_samples, dtype=complex)
        
        # Generate pulses
        pulse_width = 100  # samples
        pulse_period = 500  # samples
        
        for i in range(0, num_samples, pulse_period):
            if i + pulse_width < num_samples:
                # Random frequency for each pulse
                freq = np.random.uniform(-300e3, 300e3)
                t = np.arange(pulse_width) / self.sample_rate
                pulse = 3.0 * np.exp(1j * 2 * np.pi * freq * t)
                signal[i:i+pulse_width] = pulse
        
        # Add noise
        noise = 0.1 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        signal = signal + noise
        
        return np.real(signal), np.imag(signal)
    
    def generate_barrage_jammer(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate barrage (wideband noise) jamming signal."""
        # High power white noise
        jammer = 2.0 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        # Add some structure (filtered noise)
        from scipy import signal
        b, a = signal.butter(4, [0.2, 0.8], 'band')
        jammer_real = signal.filtfilt(b, a, np.real(jammer))
        jammer_imag = signal.filtfilt(b, a, np.imag(jammer))
        
        return jammer_real, jammer_imag
    
    def send_packet(self, i_data: np.ndarray, q_data: np.ndarray):
        """Send data packet via UDP."""
        timestamp = time.time()
        
        # Pack using protocol
        packet = RFProtocol.pack_iq_packet(timestamp, i_data, q_data)
        
        # Send
        self.socket.sendto(packet, (self.target_host, self.target_port))
    
    def run_simulation(self, duration: float = 60.0):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration in seconds
        """
        self.running = True
        start_time = time.time()
        packet_count = 0
        
        logger.info(f"Starting simulation for {duration} seconds")
        
        # Simulation schedule
        schedule = [
            (0, 10, 'normal'),      # 0-10s: normal
            (10, 15, 'tone'),       # 10-15s: tone jammer
            (15, 25, 'normal'),     # 15-25s: normal
            (25, 30, 'sweep'),      # 25-30s: sweep jammer
            (30, 35, 'normal'),     # 30-35s: normal
            (35, 40, 'pulse'),      # 35-40s: pulse jammer
            (40, 45, 'normal'),     # 40-45s: normal
            (45, 50, 'barrage'),    # 45-50s: barrage jammer
            (50, 60, 'normal'),     # 50-60s: normal
        ]
        
        while self.running and (time.time() - start_time) < duration:
            current_time = time.time() - start_time
            
            # Determine current mode from schedule
            current_mode = 'normal'
            for start, end, mode in schedule:
                if start <= current_time < end:
                    current_mode = mode
                    break
            
            # Log mode changes
            if current_mode != self.current_mode:
                logger.info(f"Switching to {current_mode} mode at {current_time:.1f}s")
                self.current_mode = current_mode
            
            # Generate appropriate signal
            if current_mode == 'normal':
                i_data, q_data = self.generate_normal_signal(self.packet_size)
            elif current_mode == 'tone':
                i_data, q_data = self.generate_tone_jammer(self.packet_size)
            elif current_mode == 'sweep':
                i_data, q_data = self.generate_sweep_jammer(self.packet_size)
            elif current_mode == 'pulse':
                i_data, q_data = self.generate_pulse_jammer(self.packet_size)
            elif current_mode == 'barrage':
                i_data, q_data = self.generate_barrage_jammer(self.packet_size)
            else:
                i_data, q_data = self.generate_normal_signal(self.packet_size)
            
            # Send packet
            self.send_packet(i_data, q_data)
            packet_count += 1
            
            # Sleep to maintain data rate
            # Approximate rate: (packet_size samples) / (sample_rate Hz) = packet duration
            packet_duration = self.packet_size / self.sample_rate
            time.sleep(packet_duration * 0.9)  # 90% to account for processing time
        
        self.running = False
        elapsed = time.time() - start_time
        logger.info(f"Simulation completed: {packet_count} packets sent in {elapsed:.1f}s")
    
    def stop(self):
        """Stop the simulation."""
        self.running = False


def test_components():
    """Test individual components."""
    import torch
    from signal_filters import SignalPreprocessor
    from autoencoder import ImprovedRFAutoencoder
    from threshold_manager import AdaptiveThresholdManager
    
    print("Testing components...")
    
    # Test signal preprocessor
    print("\n1. Testing SignalPreprocessor...")
    preprocessor = SignalPreprocessor()
    
    # Generate test signal
    i_data = np.random.randn(1024)
    q_data = np.random.randn(1024)
    
    # Test preprocessing
    tensor = preprocessor.preprocess_iq(i_data, q_data)
    print(f"   Preprocessed tensor shape: {tensor.shape}")
    
    # Test feature extraction
    features = preprocessor.extract_spectral_features(i_data, q_data)
    print(f"   Extracted {len(features)} spectral features")
    
    # Test autoencoder
    print("\n2. Testing Autoencoder...")
    model = ImprovedRFAutoencoder(input_size=1024)
    
    # Test forward pass
    with torch.no_grad():
        reconstruction, mu, logvar = model(tensor)
        anomaly_score = model.get_anomaly_score(tensor)
    
    print(f"   Reconstruction shape: {reconstruction.shape}")
    print(f"   Anomaly score: {anomaly_score.item():.3f}")
    
    # Test threshold manager
    print("\n3. Testing Threshold Manager...")
    threshold_mgr = AdaptiveThresholdManager()
    
    # Add some scores
    for i in range(100):
        score = np.random.exponential(1.0)  # Simulate scores
        threshold_mgr.update(score)
    
    stats = threshold_mgr.get_statistics()
    print(f"   Current threshold: {stats['threshold']:.3f}")
    print(f"   Mean score: {stats['mean']:.3f}")
    print(f"   Samples processed: {stats['samples_count']}")
    
    print("\nComponent tests completed successfully!")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RF Anomaly Detection System")
    parser.add_argument('--host', default='localhost', help='Target host (default: localhost)')
    parser.add_argument('--port', type=int, default=12345, help='Target port (default: 12345)')
    parser.add_argument('--duration', type=float, default=60, help='Test duration in seconds (default: 60)')
    parser.add_argument('--test-components', action='store_true', help='Test individual components')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.test_components:
        test_components()
    else:
        # Run simulation
        print(f"Starting RF data simulation...")
        print(f"Target: {args.host}:{args.port}")
        print(f"Duration: {args.duration} seconds")
        print("\nSchedule:")
        print("  0-10s:  Normal signal")
        print("  10-15s: Tone jammer")
        print("  15-25s: Normal signal")
        print("  25-30s: Sweep jammer")
        print("  30-35s: Normal signal")
        print("  35-40s: Pulse jammer")
        print("  40-45s: Normal signal")
        print("  45-50s: Barrage jammer")
        print("  50-60s: Normal signal")
        print("\nPress Ctrl+C to stop early")
        
        simulator = RFDataSimulator(args.host, args.port)
        
        try:
            simulator.run_simulation(args.duration)
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
            simulator.stop()


if __name__ == "__main__":
    main()