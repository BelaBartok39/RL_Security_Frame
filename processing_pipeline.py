"""
Main processing pipeline that orchestrates all components.
Handles data flow from receiver through detection to channel hopping.
"""

import threading
import queue
import time
import numpy as np
from typing import Optional, Dict, Callable
import logging
from dataclasses import dataclass

from receiver import RFDataReceiver
from signal_filters import SignalPreprocessor
from anomaly_detector import AnomalyDetector, AnomalyResult
from jammer_classifier import JammerClassifier
from channel_scanner import ChannelScanner
from config import SystemConfig


logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Complete result of processing pipeline."""
    timestamp: float
    anomaly_result: Optional[AnomalyResult] = None
    classification_result: Optional['ClassificationResult'] = None
    features: Optional[Dict[str, float]] = None
    action_taken: Optional[str] = None


class RFProcessingPipeline:
    """
    Main processing pipeline coordinating all components.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize processing pipeline.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.running = False
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        
        # Receiver
        self.receiver = RFDataReceiver(
            port=config.receiver.port,
            buffer_size=config.receiver.buffer_size,
            bind_address=config.receiver.bind_address,
            callback=self._on_data_received
        )
        
        # Preprocessor
        self.preprocessor = SignalPreprocessor({
            'highpass_cutoff': config.preprocessing.highpass_cutoff,
            'lowpass_cutoff': config.preprocessing.lowpass_cutoff,
            'filter_order': config.preprocessing.filter_order,
            'normalization': config.preprocessing.normalization,
            'clip_sigma': config.preprocessing.clip_sigma,
            'device': config.anomaly_detection.device
        })
        
        # Anomaly detector
        self.anomaly_detector = AnomalyDetector(config.anomaly_detection)
        
        # Jammer classifier
        self.jammer_classifier = JammerClassifier(config.classifier)
        
        # Channel scanner
        self.channel_scanner = ChannelScanner(config.channel_scanner)
        
        # Processing queue and thread
        self.processing_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        
        # Window buffer for signal processing
        self.window_size = config.preprocessing.window_size
        self.overlap = config.preprocessing.overlap
        self.signal_buffer = []
        
        # Metrics
        self.packets_processed = 0
        self.anomalies_detected = 0
        self.channel_hops = 0
        self.processing_time_avg = 0
        self.last_metrics_update = time.time()
        
        # Callbacks for GUI updates
        self.anomaly_callback: Optional[Callable] = None
        self.metrics_callback: Optional[Callable] = None
        self.signal_callback: Optional[Callable] = None
        
        logger.info("Pipeline initialization complete")
    
    def start(self):
        """Start the processing pipeline."""
        if not self.running:
            self.running = True
            
            # Start receiver
            self.receiver.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop, 
                daemon=True
            )
            self.processing_thread.start()
            
            logger.info("Processing pipeline started")
    
    def stop(self):
        """Stop the processing pipeline."""
        self.running = False
        
        # Stop receiver
        self.receiver.stop()
        
        # Wait for processing thread
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Processing pipeline stopped")
    
    def _on_data_received(self, packet_data: Dict):
        """
        Callback for received data packets.
        
        Args:
            packet_data: Dictionary with timestamp, samples, num_samples
        """
        try:
            # Add to processing queue
            self.processing_queue.put_nowait(packet_data)
        except queue.Full:
            # Drop oldest if queue is full
            try:
                self.processing_queue.get_nowait()
                self.processing_queue.put_nowait(packet_data)
            except queue.Empty:
                pass
    
    def _processing_loop(self):
        """Main processing loop."""
        logger.info("Processing loop started")
        
        while self.running:
            try:
                # Get data from queue with timeout
                packet_data = self.processing_queue.get(timeout=0.1)
                
                # Process the packet
                self._process_packet(packet_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
        
        logger.info("Processing loop stopped")
    
    def _process_packet(self, packet_data: Dict):
        """
        Process a single data packet through the pipeline.
        
        Args:
            packet_data: Packet data from receiver
        """
        start_time = time.time()
        
        # Extract I/Q samples
        samples = packet_data['samples']
        if len(samples) == 0:
            return
        
        # Add to buffer
        self.signal_buffer.extend(samples)
        
        # Process windows when we have enough data
        while len(self.signal_buffer) >= self.window_size:
            # Extract window
            window_samples = np.array(self.signal_buffer[:self.window_size])
            
            # Split I/Q
            i_data = window_samples[:, 0]
            q_data = window_samples[:, 1]
            
            # Process window
            result = self._process_window(i_data, q_data, packet_data['timestamp'])
            
            # Update metrics
            self.packets_processed += 1
            
            # Remove processed samples (with overlap)
            advance = int(self.window_size * (1 - self.overlap))
            self.signal_buffer = self.signal_buffer[advance:]
        
        # Update processing time
        process_time = time.time() - start_time
        self.processing_time_avg = 0.9 * self.processing_time_avg + 0.1 * process_time
        
        # Update metrics periodically
        if time.time() - self.last_metrics_update > self.config.metrics_update_interval:
            self._update_metrics()
    
    def _process_window(self, i_data: np.ndarray, q_data: np.ndarray, 
                       timestamp: float) -> ProcessingResult:
        """
        Process a single window of I/Q data.
        
        Args:
            i_data: In-phase data
            q_data: Quadrature data
            timestamp: Packet timestamp
            
        Returns:
            Processing result
        """
        result = ProcessingResult(timestamp=timestamp)
        
        try:
            # 1. Preprocessing
            preprocessed = self.preprocessor.preprocess_iq(i_data, q_data)
            
            # 2. Feature extraction
            spectral_features = self.preprocessor.extract_spectral_features(i_data, q_data)
            temporal_features = self.preprocessor.extract_temporal_features(i_data, q_data)
            rf_puf_features = self.preprocessor.extract_rf_puf_features(i_data, q_data)
            
            # Send signal data to GUI for visualization
            if self.signal_callback and np.random.random() < 0.1:  # Subsample to avoid overwhelming GUI
                self.signal_callback({
                    'i_data': i_data,
                    'q_data': q_data,
                    'timestamp': timestamp
                })
            
            # Combine features
            all_features = {**spectral_features, **temporal_features, **rf_puf_features}
            result.features = all_features
            
            # 3. Anomaly detection
            anomaly_result = self.anomaly_detector.detect(preprocessed)
            result.anomaly_result = anomaly_result
            
            # 4. If anomaly detected, classify and take action
            if anomaly_result.is_anomaly:
                self.anomalies_detected += 1
                
                # Classify jammer type
                classification = self.jammer_classifier.classify(preprocessed, all_features)
                result.classification_result = classification
                
                # Notify callbacks
                if self.anomaly_callback:
                    self.anomaly_callback(result)
                
                # Take action based on confidence
                if anomaly_result.confidence > 0.7:
                    self._handle_jamming_detected(result)
            
        except Exception as e:
            logger.error(f"Window processing error: {e}")
        
        return result
    
    def _handle_jamming_detected(self, result: ProcessingResult):
        """
        Handle detected jamming by initiating channel hop.
        
        Args:
            result: Processing result with anomaly and classification
        """
        logger.warning(f"Jamming detected! Type: {result.classification_result.jammer_type}, "
                      f"Confidence: {result.classification_result.confidence:.2f}")
        
        # Mark current channel as jammed
        if self.channel_scanner.current_channel:
            self.channel_scanner.mark_channel_jammed(self.channel_scanner.current_channel)
        
        # Find clean channel
        clean_channel = self.channel_scanner.find_clean_channel()
        
        if clean_channel:
            # Hop to clean channel
            reason = f"jamming_{result.classification_result.jammer_type}"
            success = self.channel_scanner.hop_to_channel(clean_channel, reason)
            
            if success:
                self.channel_hops += 1
                result.action_taken = f"hop_to_{clean_channel}"
                logger.info(f"Successfully commanded hop to {clean_channel/1e6:.1f} MHz")
            else:
                logger.error("Failed to command channel hop")
        else:
            logger.error("No clean channels available!")
            result.action_taken = "no_clean_channel"
    
    def _update_metrics(self):
        """Update and broadcast metrics."""
        metrics = self.get_metrics()
        
        if self.metrics_callback:
            self.metrics_callback(metrics)
        
        self.last_metrics_update = time.time()
    
    def get_metrics(self) -> Dict:
        """Get comprehensive pipeline metrics."""
        return {
            'pipeline': {
                'packets_processed': self.packets_processed,
                'anomalies_detected': self.anomalies_detected,
                'channel_hops': self.channel_hops,
                'processing_time_ms': self.processing_time_avg * 1000,
                'queue_size': self.processing_queue.qsize(),
                'buffer_size': len(self.signal_buffer)
            },
            'receiver': self.receiver.get_statistics(),
            'anomaly_detector': self.anomaly_detector.get_metrics(),
            'classifier': self.jammer_classifier.get_statistics(),
            'channel_scanner': self.channel_scanner.get_channel_status_summary()
        }
    
    def set_anomaly_callback(self, callback: Callable):
        """Set callback for anomaly detection events."""
        self.anomaly_callback = callback
    
    def set_metrics_callback(self, callback: Callable):
        """Set callback for metrics updates."""
        self.metrics_callback = callback
    
    def set_signal_callback(self, callback: Callable):
        """Set callback for signal visualization."""
        self.signal_callback = callback
    
    def save_models(self, directory: str):
        """Save all trained models."""
        import os
        os.makedirs(directory, exist_ok=True)
        
        self.anomaly_detector.save_model(os.path.join(directory, 'anomaly_model.pth'))
        self.jammer_classifier.save_model(os.path.join(directory, 'classifier_model.pth'))
        logger.info(f"Models saved to {directory}")