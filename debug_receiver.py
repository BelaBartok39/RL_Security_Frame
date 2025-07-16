"""
Simple debug receiver to test UDP connectivity.
"""

import socket
import struct
import sys

def debug_receiver(port=12345):
    """Simple UDP receiver for debugging."""
    print(f"Starting debug receiver on port {port}")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', port))
    
    packet_count = 0
    while True:
        try:
            data, addr = sock.recvfrom(65536)
            packet_count += 1
            
            # Try to parse header
            if len(data) >= 20:
                timestamp, num_samples, _ = struct.unpack('!dI8s', data[:20])
                print(f"Packet {packet_count} from {addr}: {len(data)} bytes, "
                      f"{num_samples} samples, timestamp={timestamp:.3f}")
            else:
                print(f"Packet {packet_count} from {addr}: {len(data)} bytes (too small)")
                
        except KeyboardInterrupt:
            print(f"\nReceived {packet_count} packets total")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 12345
    debug_receiver(port)