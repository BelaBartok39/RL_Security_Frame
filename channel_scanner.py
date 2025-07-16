"""
Channel scanner module for finding clean channels.
Communicates with HackRF via UDP to request channel scans and changes.
"""

import socket
import json
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
import threading
import queue

from protocols import RFProtocol
from config import ChannelScannerConfig


logger = logging.getLogger(__name__)


@dataclass
class ChannelStatus:
    """Status of a single channel."""
    frequency: int
    energy_dbm: float
    noise_floor_dbm: float
    is_clean: bool
    last_scan_time: float
    jamming_detected: bool = False
    occupancy_percent: float = 0.0


class ChannelScanner:
    """
    Manages channel scanning and hopping operations.
    Communicates with HackRF to find clean channels.
    """
    
    def __init__(self, config: ChannelScannerConfig, 
                 tx_address: str = 'localhost'):
        """
        Initialize channel scanner.
        
        Args:
            config: Channel scanner configuration
            tx_address: Address of HackRF control interface
        """
        self.config = config
        self.tx_address = tx_address
        
        # Channel status tracking
        self.channel_status: Dict[int, ChannelStatus] = {}
        for freq in config.scan_channels:
            self.channel_status[freq] = ChannelStatus(
                frequency=freq,
                energy_dbm=-100.0,
                noise_floor_dbm=-100.0,
                is_clean=True,
                last_scan_time=0
            )
        
        # Current channel
        self.current_channel = config.scan_channels[0] if config.scan_channels else None
        
        # UDP socket for sending commands
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Scan results queue
        self.scan_results_queue = queue.Queue(maxsize=100)
        
        # Statistics
        self.hop_count = 0
        self.last_hop_time = None
        self.hop_history = []
        
        logger.info(f"Channel scanner initialized with {len(config.scan_channels)} channels")
    
    def request_channel_scan(self, frequency: int) -> bool:
        """
        Request HackRF to scan a specific channel.
        
        Args:
            frequency: Frequency to scan in Hz
            
        Returns:
            True if request sent successfully
        """
        try:
            # Create scan request message
            scan_msg = {
                'command': 'SCAN_CHANNEL',
                'frequency': frequency,
                'duration': self.config.scan_duration,
                'timestamp': time.time()
            }
            
            # Send to HackRF
            msg_bytes = json.dumps(scan_msg).encode()
            self.command_socket.sendto(msg_bytes, (self.tx_address, self.config.tx_port))
            
            logger.debug(f"Scan request sent for {frequency/1e6:.1f} MHz")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send scan request: {e}")
            return False
    
    def find_clean_channel(self, exclude_current: bool = True) -> Optional[int]:
        """
        Find the cleanest available channel.
        
        Args:
            exclude_current: Whether to exclude current channel
            
        Returns:
            Frequency of cleanest channel or None
        """
        # Update channel status based on recent scans
        self._update_channel_status()
        
        # Get candidate channels
        candidates = []
        current_time = time.time()
        
        for freq, status in self.channel_status.items():
            # Skip current channel if requested
            if exclude_current and freq == self.current_channel:
                continue
            
            # Skip if recently scanned and found dirty
            if not status.is_clean and (current_time - status.last_scan_time) < 10:
                continue
            
            # Add to candidates with score
            score = self._calculate_channel_score(status)
            candidates.append((freq, score, status))
        
        # Sort by score (lower is better)
        candidates.sort(key=lambda x: x[1])
        
        if candidates:
            best_freq, score, status = candidates[0]
            logger.info(f"Found clean channel: {best_freq/1e6:.1f} MHz "
                       f"(energy: {status.energy_dbm:.1f} dBm)")
            return best_freq
        
        logger.warning("No clean channels found")
        return None
    
    def _calculate_channel_score(self, status: ChannelStatus) -> float:
        """
        Calculate channel quality score (lower is better).
        
        Args:
            status: Channel status
            
        Returns:
            Quality score
        """
        score = 0.0
        
        # Energy level (lower is better)
        score += status.energy_dbm + 100  # Normalize around -100 dBm
        
        # Penalize if jamming detected
        if status.jamming_detected:
            score += 100
        
        # Penalize high occupancy
        score += status.occupancy_percent
        
        # Slight penalty for stale data
        age = time.time() - status.last_scan_time
        score += min(age / 60, 10)  # Max 10 point penalty for old data
        
        return score
    
    def hop_to_channel(self, frequency: int, reason: str = "jamming_detected") -> bool:
        """
        Command HackRF to hop to a new channel.
        
        Args:
            frequency: New frequency in Hz
            reason: Reason for channel change
            
        Returns:
            True if command sent successfully
        """
        try:
            # Create channel change message
            change_msg = RFProtocol.create_channel_change_message(frequency, reason)
            
            # Send to HackRF
            msg_bytes = json.dumps(change_msg).encode()
            self.command_socket.sendto(msg_bytes, (self.tx_address, self.config.tx_port))
            
            # Update tracking
            self.current_channel = frequency
            self.hop_count += 1
            self.last_hop_time = time.time()
            
            # Record in history
            self.hop_history.append({
                'timestamp': self.last_hop_time,
                'from_freq': self.current_channel,
                'to_freq': frequency,
                'reason': reason
            })
            
            # Keep history bounded
            if len(self.hop_history) > 1000:
                self.hop_history.pop(0)
            
            logger.info(f"Channel hop commanded: {frequency/1e6:.1f} MHz (reason: {reason})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send channel hop command: {e}")
            return False
    
    def update_channel_energy(self, frequency: int, energy_dbm: float, 
                            noise_floor_dbm: Optional[float] = None):
        """
        Update channel energy measurement.
        
        Args:
            frequency: Channel frequency
            energy_dbm: Measured energy in dBm
            noise_floor_dbm: Noise floor if available
        """
        if frequency in self.channel_status:
            status = self.channel_status[frequency]
            status.energy_dbm = energy_dbm
            if noise_floor_dbm is not None:
                status.noise_floor_dbm = noise_floor_dbm
            status.last_scan_time = time.time()
            
            # Update clean status based on threshold
            status.is_clean = energy_dbm < self.config.energy_threshold_db
    
    def mark_channel_jammed(self, frequency: int):
        """Mark a channel as jammed."""
        if frequency in self.channel_status:
            self.channel_status[frequency].jamming_detected = True
            self.channel_status[frequency].is_clean = False
            logger.warning(f"Channel {frequency/1e6:.1f} MHz marked as jammed")
    
    def _update_channel_status(self):
        """Update channel status from any pending scan results."""
        # Process any queued scan results
        while not self.scan_results_queue.empty():
            try:
                result = self.scan_results_queue.get_nowait()
                if 'frequency' in result and 'energy_dbm' in result:
                    self.update_channel_energy(
                        result['frequency'],
                        result['energy_dbm'],
                        result.get('noise_floor_dbm')
                    )
            except queue.Empty:
                break
    
    def get_channel_status_summary(self) -> Dict:
        """Get summary of all channel status."""
        clean_channels = sum(1 for s in self.channel_status.values() if s.is_clean)
        jammed_channels = sum(1 for s in self.channel_status.values() if s.jamming_detected)
        
        return {
            'total_channels': len(self.channel_status),
            'clean_channels': clean_channels,
            'jammed_channels': jammed_channels,
            'current_channel': self.current_channel,
            'hop_count': self.hop_count,
            'last_hop_time': self.last_hop_time,
            'channels': {
                freq: {
                    'energy_dbm': status.energy_dbm,
                    'is_clean': status.is_clean,
                    'jamming_detected': status.jamming_detected,
                    'last_scan': status.last_scan_time
                }
                for freq, status in self.channel_status.items()
            }
        }
    
    def get_hop_history(self, last_n: Optional[int] = None) -> List[Dict]:
        """Get channel hop history."""
        if last_n is None:
            return self.hop_history
        return self.hop_history[-last_n:]
    
    def perform_channel_sweep(self):
        """Perform a full sweep of all configured channels."""
        logger.info("Starting channel sweep")
        
        for freq in self.config.scan_channels:
            self.request_channel_scan(freq)
            time.sleep(self.config.scan_duration + 0.1)  # Allow time for scan
        
        logger.info("Channel sweep completed")