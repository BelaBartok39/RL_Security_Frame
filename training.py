"""
Training module for VAE and DRN models.
Optimized for Jetson GPU acceleration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
import time
import os
from tqdm import tqdm

from autoencoder import ImprovedRFAutoencoder
from jammer_classifier import JammerClassifierDRN
from signal_filters import SignalPreprocessor


logger = logging.getLogger(__name__)


class RFSignalDataset(Dataset):
    """Dataset for RF signal windows."""
    
    def __init__(self, data_path: str, window_size: int = 1024, 
                 label: Optional[str] = None, preprocessor: Optional[SignalPreprocessor] = None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data files or directory
            window_size: Size of signal windows
            label: Label for supervised training (jammer type)
            preprocessor: Signal preprocessor
        """
        self.window_size = window_size
        self.label = label
        self.preprocessor = preprocessor or SignalPreprocessor()
        
        # Load data
        self.samples = []
        self._load_data(data_path)
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self, data_path: str):
        """Load data from files."""
        # This is a placeholder - implement based on your data format
        # For now, let's assume numpy files with I/Q data
        if os.path.isfile(data_path):
            data = np.load(data_path)
            # Assume data is shape (n_samples, 2) for I/Q
            for i in range(0, len(data) - self.window_size, self.window_size // 2):
                window = data[i:i + self.window_size]
                self.samples.append(window)
        elif os.path.isdir(data_path):
            for file in os.listdir(data_path):
                if file.endswith('.npy'):
                    filepath = os.path.join(data_path, file)
                    data = np.load(filepath)
                    for i in range(0, len(data) - self.window_size, self.window_size // 2):
                        window = data[i:i + self.window_size]
                        self.samples.append(window)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get preprocessed sample."""
        window = self.samples[idx]
        i_data = window[:, 0]
        q_data = window[:, 1]
        
        # Preprocess
        tensor = self.preprocessor.preprocess_iq(i_data, q_data)
        
        if self.label is not None:
            return tensor, self.label
        return tensor


class ModelTrainer:
    """Handles training of VAE and DRN models with GPU optimization."""
    
    def __init__(self, device: str = 'cuda'):
        """Initialize trainer."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Check CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Enable cuDNN autotuner for Jetson optimization
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        else:
            logger.warning("CUDA not available, using CPU")
        
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    
    def train_vae(self, 
                  model: ImprovedRFAutoencoder,
                  train_loader: DataLoader,
                  val_loader: Optional[DataLoader] = None,
                  epochs: int = 50,
                  learning_rate: float = 1e-3,
                  save_path: Optional[str] = None) -> Dict:
        """
        Train VAE on normal signal data.
        
        Args:
            model: VAE model
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        logger.info(f"Starting VAE training on {self.device}")
        model = model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_batches = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Mixed precision training for Jetson
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        reconstruction, mu, logvar = model(batch)
                        loss_dict = model.loss_function(batch, reconstruction, mu, logvar)
                        loss = loss_dict['loss']
                    
                    optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    reconstruction, mu, logvar = model(batch)
                    loss_dict = model.loss_function(batch, reconstruction, mu, logvar)
                    loss = loss_dict['loss']
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / train_batches
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            if val_loader:
                model.eval()
                val_loss = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        reconstruction, mu, logvar = model(batch)
                        loss_dict = model.loss_function(batch, reconstruction, mu, logvar)
                        val_loss += loss_dict['loss'].item()
                        val_batches += 1
                
                avg_val_loss = val_loss / val_batches
                history['val_loss'].append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Save best model
                if save_path and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_val_loss,
                    }, save_path)
                    logger.info(f"Saved best model with val_loss: {avg_val_loss:.4f}")
                
                logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                          f"val_loss={avg_val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
            else:
                logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}")
            
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and epoch % 10 == 0:
                torch.cuda.empty_cache()
        
        return history
    
    def train_classifier(self,
                        model: JammerClassifierDRN,
                        train_loader: DataLoader,
                        val_loader: Optional[DataLoader] = None,
                        epochs: int = 30,
                        learning_rate: float = 1e-3,
                        save_path: Optional[str] = None) -> Dict:
        """
        Train DRN classifier for jammer type identification.
        
        Args:
            model: DRN classifier model
            train_loader: Training data loader with labels
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        logger.info(f"Starting DRN classifier training on {self.device}")
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Mixed precision training
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item(), 
                                 'acc': 100 * train_correct / train_total})
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            
            # Validation phase
            if val_loader:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100 * val_correct / val_total
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_accuracy)
                
                # Save best model
                if save_path and val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': val_accuracy,
                    }, save_path)
                    logger.info(f"Saved best model with val_acc: {val_accuracy:.2f}%")
                
                logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                          f"train_acc={train_accuracy:.2f}%, "
                          f"val_loss={avg_val_loss:.4f}, "
                          f"val_acc={val_accuracy:.2f}%")
            else:
                logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
                          f"train_acc={train_accuracy:.2f}%")
            
            scheduler.step()
            
            # Clear GPU cache
            if torch.cuda.is_available() and epoch % 10 == 0:
                torch.cuda.empty_cache()
        
        return history


def collect_normal_data(duration: int = 60, save_path: str = 'normal_data.npy'):
    """
    Collect normal RF data for VAE training.
    
    Args:
        duration: Collection duration in seconds
        save_path: Path to save collected data
    """
    from receiver import RFDataReceiver
    
    logger.info(f"Collecting normal data for {duration} seconds...")
    
    collected_data = []
    
    def on_data(packet_data):
        collected_data.extend(packet_data['samples'])
    
    receiver = RFDataReceiver(callback=on_data)
    receiver.start()
    
    time.sleep(duration)
    
    receiver.stop()
    
    # Save collected data
    data_array = np.array(collected_data)
    np.save(save_path, data_array)
    logger.info(f"Saved {len(data_array)} samples to {save_path}")


if __name__ == "__main__":
    # Example training script
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RF anomaly detection models")
    parser.add_argument('--collect', action='store_true', help='Collect normal data')
    parser.add_argument('--train-vae', action='store_true', help='Train VAE model')
    parser.add_argument('--train-drn', action='store_true', help='Train DRN classifier')
    parser.add_argument('--data-path', default='./data', help='Path to training data')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.collect:
        collect_normal_data(duration=120, save_path='normal_data.npy')
    
    if args.train_vae:
        # Create dataset and dataloader
        dataset = RFSignalDataset(args.data_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
        
        # Initialize model and trainer
        model = ImprovedRFAutoencoder()
        trainer = ModelTrainer(device=args.device)
        
        # Train
        history = trainer.train_vae(model, dataloader, epochs=args.epochs,
                                   save_path='models/vae_model.pth')
        
        print("VAE training completed!")
    
    if args.train_drn:
        # Would need labeled data for different jammer types
        print("DRN training requires labeled jammer data - not implemented in this example")