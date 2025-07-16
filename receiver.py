"""
UDP packet receiver for RF data.
Handles real-time data reception and buffering.
"""

import socket
import struct
import threading
import queue
import numpy as np
from typing import Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class RFDataReceiver:
    """
    Receives RF I/Q data over UDP and manages buffering.
    """
    
    def __init__(self, 
                 port: int = 12345,
                 buffer_size: int = 65536,
                 bind_address: str = '0.0.0.0',
                 callback: Optional[Callable] = None):
        """
        Initialize receiver.
        
        Args:
            port: UDP port to listen on
            buffer_size: UDP buffer size
            bind_address: Address to bind to
            callback: Optional callback for received data
        """
        self.port = port
        self.buffer_size = buffer_size
        self.bind_address = bind_address
        self.callback = callback
        
        # Socket setup
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
        self.socket.bind((bind_address, port))
        self.socket.settimeout(0.1)
        
        # Threading
        self.running = False
        self.receive_thread = None
        
        # Data queue
        self.data_queue = queue.Queue(maxsize=1000)
        
        # Statistics
        self.packets_received = 0
        self.bytes_received = 0
        self.last_timestamp = 0
        
        logger.info(f"Receiver initialized on {bind_address}:{port}")
    
    def start(self):
        """Start receiving data."""
        if not self.running:
            self.running = True
            self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.receive_thread.start()
            logger.info("Receiver started")
    
    def stop(self):
        """Stop receiving data."""
        self.running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=1.0)
        logger.info("Receiver stopped")
    
    def _receive_loop(self):
        """Main receive loop."""
        print(f"Starting receive loop on port {self.port}")
        while self.running:
            try:
                data, addr = self.socket.recvfrom(self.buffer_size)
                print(f"Received {len(data)} bytes from {addr}")
                
                if len(data) >= 20:  # Minimum packet size
                    self._process_packet(data)
                else:
                    print(f"Packet too small: {len(data)} bytes")
                    
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Receive error: {e}")
                logger.error(f"Receive error: {e}")
    
    def _process_packet(self, data: bytes):
        """
        Process received UDP packet.
        
        Expected packet format:
        - 8 bytes: timestamp (double)
        - 4 bytes: number of samples (uint32)
        - 8 bytes: reserved
        - N*8 bytes: I/Q samples (float32 pairs)
        """
        try:
            # Parse header
            timestamp = struct.unpack('!d', data[0:8])[0]
            num_samples = struct.unpack('!I', data[8:12])[0]
            
            # Extract I/Q samples
            samples = []
            offset = 20
            
            for i in range(num_samples):
                if offset + 8 > len(data):
                    break
                    
                i_val = struct.unpack('!f', data[offset:offset+4])[0]
                q_val = struct.unpack('!f', data[offset+4:offset+8])[0]
                samples.append((i_val, q_val))
                offset += 8
            
            # Update statistics
            self.packets_received += 1
            self.bytes_received += len(data)
            self.last_timestamp = timestamp
            
            # Package data
            packet_data = {
                'timestamp': timestamp,
                'samples': np.array(samples),
                'num_samples': len(samples)
            }
            
            # Add to queue
            try:
                self.data_queue.put_nowait(packet_data)
            except queue.Full:
                # Drop oldest packet
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(packet_data)
                except queue.Empty:
                    pass
            
            # Call callback if provided
            if self.callback:
                print(f"Calling callback with {len(samples)} samples")
                self.callback(packet_data)
            else:
                print("No callback defined")
                
        except Exception as e:
            print(f"Packet processing error: {e}")
            logger.error(f"Packet processing error: {e}")
    
    def get_data(self, timeout: float = 0.1) -> Optional[dict]:
        """
        Get data from queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Packet data or None
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_statistics(self) -> dict:
        """Get receiver statistics."""
        return {
            'packets_received': self.packets_received,
            'bytes_received': self.bytes_received,
            'last_timestamp': self.last_timestamp,
            'queue_size': self.data_queue.qsize()
        }
    
    def __del__(self):
        """Cleanup."""
        try:
            self.stop()
            self.socket.close()
        except:
            pass