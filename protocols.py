"""
Network protocol definitions for RF data transmission.
"""

import struct
from typing import Tuple, List, NamedTuple
import numpy as np


class PacketHeader(NamedTuple):
    """UDP packet header structure."""
    timestamp: float
    num_samples: int
    reserved: bytes


class ChannelChangeMessage(NamedTuple):
    """Channel change command structure."""
    command: str
    frequency: int
    timestamp: float
    reason: str


class RFProtocol:
    """
    Protocol definitions for RF data communication.
    """
    
    # Packet format constants
    HEADER_SIZE = 20
    SAMPLE_SIZE = 8  # 4 bytes I + 4 bytes Q
    MAX_SAMPLES_PER_PACKET = 1024
    
    # Command types
    CMD_CHANNEL_CHANGE = "CHANNEL_CHANGE"
    CMD_STATUS_REQUEST = "STATUS_REQUEST"
    CMD_STATUS_RESPONSE = "STATUS_RESPONSE"
    
    @staticmethod
    def pack_iq_packet(timestamp: float, i_data: np.ndarray, q_data: np.ndarray) -> bytes:
        """
        Pack I/Q data into UDP packet.
        
        Args:
            timestamp: Packet timestamp
            i_data: In-phase samples
            q_data: Quadrature samples
            
        Returns:
            Packed byte string
        """
        num_samples = min(len(i_data), RFProtocol.MAX_SAMPLES_PER_PACKET)
        
        # Pack header
        header = struct.pack('!dI8s', timestamp, num_samples, b'\x00' * 8)
        
        # Pack samples
        samples = b''
        for i in range(num_samples):
            samples += struct.pack('!ff', float(i_data[i]), float(q_data[i]))
        
        return header + samples
    
    @staticmethod
    def unpack_iq_packet(data: bytes) -> Tuple[PacketHeader, np.ndarray, np.ndarray]:
        """
        Unpack I/Q data from UDP packet.
        
        Args:
            data: Packed byte string
            
        Returns:
            Tuple of (header, i_data, q_data)
        """
        # Unpack header
        timestamp, num_samples, reserved = struct.unpack('!dI8s', data[:20])
        header = PacketHeader(timestamp, num_samples, reserved)
        
        # Unpack samples
        i_data = []
        q_data = []
        offset = 20
        
        for _ in range(num_samples):
            if offset + 8 <= len(data):
                i_val, q_val = struct.unpack('!ff', data[offset:offset+8])
                i_data.append(i_val)
                q_data.append(q_val)
                offset += 8
        
        return header, np.array(i_data), np.array(q_data)
    
    @staticmethod
    def create_channel_change_message(frequency: int, reason: str = "jamming_detected") -> dict:
        """
        Create channel change message.
        
        Args:
            frequency: New frequency in Hz
            reason: Reason for channel change
            
        Returns:
            Message dictionary
        """
        import time
        
        return {
            'command': RFProtocol.CMD_CHANNEL_CHANGE,
            'frequency': frequency,
            'timestamp': time.time(),
            'reason': reason
        }