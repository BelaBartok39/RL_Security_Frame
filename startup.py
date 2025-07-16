"""
Startup script for RF anomaly detection system.
Handles initial training and system initialization.
"""

import os
import sys
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import SystemConfig
from processing_pipeline import RFProcessingPipeline
from training import RFSignalDataset, ModelTrainer, collect_normal_data
from gui import RFMonitoringGUI


logger = logging.getLogger(__name__)


def check_gpu_status():
    """Check and report GPU status."""
    if torch.cuda.is_available():
        logger.info("=" * 50)
        logger.info("GPU INFORMATION:")
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch CUDA: {torch.cuda.is_available()}")
        logger.info("=" * 50)
        return True
    else:
        logger.warning("GPU not available! System will run on CPU (slow)")
        return False


def initialize_or_train_vae(pipeline, config):
    """Initialize VAE with existing model or train new one."""
    model_path = config.anomaly_detection.model_path or 'models/vae_model.pth'
    
    if os.path.exists(model_path):
        logger.info(f"Loading existing VAE model from {model_path}")
        pipeline.anomaly_detector.load_model(model_path)
        return
    
    logger.info("No trained VAE model found. Starting training process...")
    
    # Check for training data
    normal_data_path = 'data/normal_data.npy'
    if not os.path.exists(normal_data_path):
        logger.info("No training data found. Collecting normal signal data...")
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Collect normal data
        print("\n" + "="*60)
        print("COLLECTING NORMAL RF DATA FOR TRAINING")
        print("Please ensure RF receiver is connected and receiving clean signals")
        print("Collection will run for 2 minutes...")
        print("="*60 + "\n")
        
        collect_normal_data(duration=120, save_path=normal_data_path, 
                           receiver=pipeline.receiver)
    
    # Create dataset and train
    logger.info("Creating training dataset...")
    dataset = RFSignalDataset(normal_data_path, window_size=config.preprocessing.window_size)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                            num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                          num_workers=2, pin_memory=True)
    
    # Train model
    trainer = ModelTrainer(device=config.anomaly_detection.device)
    
    print("\n" + "="*60)
    print("TRAINING VAE MODEL")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Device: {config.anomaly_detection.device}")
    print("="*60 + "\n")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train
    history = trainer.train_vae(
        pipeline.anomaly_detector.model,
        train_loader,
        val_loader,
        epochs=30,  # Reduced for faster initial training
        learning_rate=1e-3,
        save_path=model_path
    )
    
    # Reload best model
    pipeline.anomaly_detector.load_model(model_path)
    logger.info("VAE training completed and model loaded")


def run_system_diagnostics(pipeline):
    """Run system diagnostics and report status."""
    logger.info("\n" + "="*60)
    logger.info("SYSTEM DIAGNOSTICS")
    logger.info("="*60)
    
    # Test receiver
    logger.info("Testing UDP receiver...")
    stats = pipeline.receiver.get_statistics()
    logger.info(f"Receiver status: Ready on port {pipeline.config.receiver.port}")
    
    # Test anomaly detector
    logger.info("Testing anomaly detector...")
    test_signal = torch.randn(1, 2, 1024).to(pipeline.anomaly_detector.device)
    with torch.no_grad():
        score = pipeline.anomaly_detector.model.get_anomaly_score(test_signal)
    logger.info(f"Anomaly detector status: OK (test score: {score.item():.3f})")
    
    # Test classifier
    logger.info("Testing jammer classifier...")
    logger.info(f"Classifier status: OK ({len(pipeline.jammer_classifier.jammer_types)} types)")
    
    # Channel scanner
    logger.info("Testing channel scanner...")
    logger.info(f"Channel scanner: {len(pipeline.channel_scanner.channel_status)} channels configured")
    
    logger.info("="*60 + "\n")


def main():
    """Main startup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RF Anomaly Detection System Startup")
    parser.add_argument('--config', default='config.json', help='Configuration file')
    parser.add_argument('--train-only', action='store_true', help='Only train models and exit')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    parser.add_argument('--skip-training', action='store_true', help='Skip training phase')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("RF ANOMALY DETECTION SYSTEM")
    print("Optimized for NVIDIA Jetson")
    print("="*60 + "\n")
    
    # Check GPU
    gpu_available = check_gpu_status()
    
    # Load configuration
    if os.path.exists(args.config):
        config = SystemConfig.from_file(args.config)
    else:
        config = SystemConfig()
        config.save(args.config)
    
    # Force GPU usage if available
    if gpu_available and config.anomaly_detection.device != 'cuda':
        logger.info("Forcing GPU usage for optimal performance")
        config.anomaly_detection.device = 'cuda'
    
    try:
        # Initialize pipeline
        logger.info("Initializing processing pipeline...")
        pipeline = RFProcessingPipeline(config)
        
        # Initialize or train VAE
        if not args.skip_training:
            initialize_or_train_vae(pipeline, config)
        
        if args.train_only:
            logger.info("Training completed. Exiting...")
            return 0
        
        # Run diagnostics
        run_system_diagnostics(pipeline)
        
        # Start system
        if args.no_gui:
            logger.info("Starting in headless mode...")
            pipeline.start()
            
            try:
                import time
                while True:
                    time.sleep(5)
                    metrics = pipeline.get_metrics()
                    logger.info(f"Status - Packets: {metrics['pipeline']['packets_processed']}, "
                              f"Anomalies: {metrics['pipeline']['anomalies_detected']}")
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                pipeline.stop()
        else:
            logger.info("Starting GUI...")
            gui = RFMonitoringGUI(pipeline)
            pipeline.start()
            
            print("\n" + "="*60)
            print("SYSTEM READY")
            print("GUI is running. Use menu options to control the system.")
            print("="*60 + "\n")
            
            gui.run()
            pipeline.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())