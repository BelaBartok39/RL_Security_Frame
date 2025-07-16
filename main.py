"""
Main application entry point for RF anomaly detection system.
Handles initialization, configuration, and launching the GUI.
"""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from typing import Optional

from config import SystemConfig
from processing_pipeline import RFProcessingPipeline
from gui import RFMonitoringGUI


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )
    
    # Suppress matplotlib debug messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # Suppress PIL debug messages
    logging.getLogger('PIL').setLevel(logging.WARNING)


def create_default_config_file(filepath: str):
    """Create a default configuration file."""
    config = SystemConfig()
    config.save(filepath)
    print(f"Created default configuration file: {filepath}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="RF Anomaly Detection System - Jamming Detection and Mitigation"
    )
    
    # Configuration arguments
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.json',
        help='Configuration file path (default: config.json)'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file and exit'
    )
    
    # Logging arguments
    parser.add_argument(
        '-l', '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (optional)'
    )
    
    # Mode arguments
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run without GUI (headless mode)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with simulated data'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Create default config if requested
    if args.create_config:
        create_default_config_file(args.config)
        return 0
    
    # Load configuration
    try:
        if os.path.exists(args.config):
            config = SystemConfig.from_file(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.warning(f"Configuration file not found: {args.config}")
            logger.info("Using default configuration")
            config = SystemConfig()
            
            # Save default config for future use
            config.save(args.config)
            logger.info(f"Saved default configuration to {args.config}")
    
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Update model paths if specified
    if args.model_dir:
        config.anomaly_detection.model_path = os.path.join(args.model_dir, 'anomaly_model.pth')
        config.classifier.model_path = os.path.join(args.model_dir, 'classifier_model.pth')
    
    # Create anomaly save directory
    os.makedirs(config.anomaly_save_path, exist_ok=True)
    
    try:
        # Initialize processing pipeline
        logger.info("Initializing RF processing pipeline...")
        pipeline = RFProcessingPipeline(config)
        
        if args.test_mode:
            logger.info("Running in test mode - simulated data will be used")
            # TODO: Implement test data generator
        
        if args.no_gui:
            # Headless mode
            logger.info("Running in headless mode")
            
            # Start pipeline
            pipeline.start()
            
            # Run until interrupted
            try:
                import time
                while True:
                    time.sleep(1)
                    
                    # Print metrics periodically
                    metrics = pipeline.get_metrics()
                    logger.info(f"Packets: {metrics['pipeline']['packets_processed']}, "
                              f"Anomalies: {metrics['pipeline']['anomalies_detected']}, "
                              f"Hops: {metrics['pipeline']['channel_hops']}")
                    
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                pipeline.stop()
        
        else:
            # GUI mode
            logger.info("Starting GUI...")
            
            # Create and run GUI
            gui = RFMonitoringGUI(pipeline)
            
            # Start pipeline
            pipeline.start()
            
            # Run GUI main loop
            gui.run()
            
            # Cleanup
            pipeline.stop()
        
        logger.info("Application terminated successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())