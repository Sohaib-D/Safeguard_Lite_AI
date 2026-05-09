"""
Real-time Packet Capture Engine for Safeguard-AI IDS

Architecture:
- AsyncSniffer for cross-platform packet capture
- asyncio.Queue for packet buffering
- Feature extraction with statistical aggregation
- Detection hooks for threat identification
- Event emission for real-time processing
"""

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import threading
import time

from scapy.all import AsyncSniffer, IP, TCP, UDP, ICMP, DNS
import pyshark
import numpy as np
from pydantic import BaseModel

from backend.services.detection_engine import DetectionEngine

logger = logging.getLogger(__name__)

@dataclass
class PacketEvent:
    """Represents a captured packet with metadata"""
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: str
    length: int
    flags: Optional[str] = None
    payload_size: int = 0
    raw_packet: Any = None


class PacketQueue:
    """Async-safe packet queue with size limits"""

    def __init__(self, max_size: int = 10000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self._lock = asyncio.Lock()

    async def put(self, packet: PacketEvent):
        async with self._lock:
            if self.queue.full():
                # Drop oldest packet
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await self.queue.put(packet)

    async def get(self) -> PacketEvent:
        return await self.queue.get()

class FeatureExtractor:
    """Extracts features from packets for detection"""

    def __init__(self, window_size: int = 60):  # 60 second window
        self.window_size = window_size
        self.packets = deque(maxlen=10000)  # Recent packets
        self.flows: Dict[tuple, Dict] = {}  # (src_ip, dst_ip, src_port, dst_port, proto) -> stats
        self.ip_stats: Dict[str, Dict] = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'ports_scanned': set(),
            'syn_count': 0,
            'last_seen': datetime.now()
        })

    def extract_features(self, packet: PacketEvent) -> Dict[str, Any]:
        """Extract features from a single packet"""
        features = {
            'src_ip': packet.src_ip,
            'dst_ip': packet.dst_ip,
            'protocol': packet.protocol,
            'packet_length': packet.length,
            'payload_size': packet.payload_size,
            'is_syn': packet.flags == 'S' if packet.flags else False,
            'is_ack': 'A' in (packet.flags or ''),
            'is_rst': 'R' in (packet.flags or ''),
            'is_fin': 'F' in (packet.flags or ''),
            'src_port': packet.src_port,
            'dst_port': packet.dst_port,
            'timestamp': packet.timestamp
        }

        # Update flow statistics
        flow_key = (packet.src_ip, packet.dst_ip, packet.src_port, packet.dst_port, packet.protocol)
        if flow_key not in self.flows:
            self.flows[flow_key] = {
                'start_time': packet.timestamp,
                'packet_count': 0,
                'byte_count': 0,
                'duration': 0
            }

        flow = self.flows[flow_key]
        flow['packet_count'] += 1
        flow['byte_count'] += packet.length
        flow['duration'] = (packet.timestamp - flow['start_time']).total_seconds()

        # Update IP statistics
        ip_stat = self.ip_stats[packet.src_ip]
        ip_stat['packet_count'] += 1
        ip_stat['byte_count'] += packet.length
        ip_stat['last_seen'] = packet.timestamp

        if packet.protocol == 'TCP' and features['is_syn'] and not features['is_ack']:
            ip_stat['syn_count'] += 1

        if packet.dst_port:
            ip_stat['ports_scanned'].add(packet.dst_port)

        # Calculate rates (packets per second)
        time_window = timedelta(seconds=self.window_size)
        recent_packets = [p for p in self.packets if packet.timestamp - p.timestamp < time_window]
        features['packet_rate'] = len(recent_packets) / self.window_size if recent_packets else 0

        # SYN flood indicator
        syn_packets = [p for p in recent_packets if getattr(p, 'flags', None) == 'S']
        features['syn_rate'] = len(syn_packets) / self.window_size

        # Port scan indicator
        features['unique_ports'] = len(ip_stat['ports_scanned'])

        # DNS tunneling indicators
        if packet.protocol == 'UDP' and packet.dst_port == 53 and packet.payload_size > 512:
            features['large_dns_query'] = True
        else:
            features['large_dns_query'] = False

        # Beaconing detection (periodic connections)
        if packet.protocol == 'TCP' and features['is_syn']:
            # Check for regular intervals (simplified)
            features['connection_interval'] = self._calculate_connection_interval(packet.src_ip, packet.timestamp)

        self.packets.append(packet)

        # Clean old flows
        cutoff = packet.timestamp - time_window
        self.flows = {k: v for k, v in self.flows.items() if v['start_time'] > cutoff}

        return features

    def _calculate_connection_interval(self, src_ip: str, timestamp: datetime) -> float:
        """Calculate average connection interval for beaconing detection"""
        connections = [p.timestamp for p in self.packets
                      if p.src_ip == src_ip and getattr(p, 'flags', None) == 'S']
        if len(connections) < 2:
            return 0.0

        intervals = [(connections[i] - connections[i-1]).total_seconds()
                    for i in range(1, len(connections))]
        return np.mean(intervals) if intervals else 0.0


class PacketSniffer:
    """Main packet capture service"""

    def __init__(self, interface: str = None, queue: PacketQueue = None,
                 extractor: FeatureExtractor = None, detector: Any = None):
        self.interface = interface or self._get_default_interface()
        self.queue = queue or PacketQueue()
        self.extractor = extractor or FeatureExtractor()
        if detector is None:
            raise ValueError("PacketSniffer requires a detector implementation")
        self.detector = detector
        self.sniffer: Optional[AsyncSniffer] = None
        self.running = False
        self.event_callbacks: List[Callable] = []

    def _get_default_interface(self) -> str:
        """Get default network interface"""
        from scapy.all import get_if_list
        interfaces = get_if_list()
        # Prefer non-loopback interface
        for iface in interfaces:
            if not iface.startswith('lo'):
                return iface
        return interfaces[0] if interfaces else 'eth0'

    def add_event_callback(self, callback: Callable):
        """Add callback for detection events"""
        self.event_callbacks.append(callback)

    async def _packet_handler(self, packet):
        """Handle captured packet"""
        try:
            packet_event = self._parse_packet(packet)
            if packet_event:
                await self.queue.put(packet_event)
        except Exception as e:
            logger.error(f"Packet handler error: {e}")

    def _parse_packet(self, packet) -> Optional[PacketEvent]:
        """Parse scapy packet into PacketEvent"""
        if IP not in packet:
            return None

        ip = packet[IP]
        src_ip = ip.src
        dst_ip = ip.dst
        protocol = 'UNKNOWN'
        src_port = dst_port = None
        flags = None

        if TCP in packet:
            protocol = 'TCP'
            tcp = packet[TCP]
            src_port = tcp.sport
            dst_port = tcp.dport
            flags = tcp.flags
        elif UDP in packet:
            protocol = 'UDP'
            udp = packet[UDP]
            src_port = udp.sport
            dst_port = udp.dport
        elif ICMP in packet:
            protocol = 'ICMP'

        return PacketEvent(
            timestamp=datetime.now(),
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            length=len(packet),
            flags=str(flags) if flags else None,
            payload_size=len(packet.payload) if hasattr(packet, 'payload') else 0,
            raw_packet=packet
        )

    async def start_capture(self):
        """Start packet capture"""
        if self.running:
            return

        self.running = True
        logger.info(f"Starting packet capture on interface {self.interface}")

        # Start processing worker
        asyncio.create_task(self._processing_worker())

        # Start sniffer
        self.sniffer = AsyncSniffer(
            iface=self.interface,
            prn=self._packet_handler,
            store=0,  # Don't store packets
            filter="ip"  # Only IP packets
        )
        await self.sniffer.start()

    async def stop_capture(self):
        """Stop packet capture"""
        self.running = False
        if self.sniffer:
            await self.sniffer.stop()
        logger.info("Packet capture stopped")

    async def _processing_worker(self):
        """Background worker for processing packets"""
        while self.running:
            try:
                packet = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                features = self.extractor.extract_features(packet)
                detections = await self.detector.detect(features)

                # Emit events
                for detection in detections:
                    for callback in self.event_callbacks:
                        try:
                            await callback(detection)
                        except Exception as e:
                            logger.error(f"Event callback error: {e}")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Processing worker error: {e}")

class PacketCaptureService:
    """High-level service for packet capture operations"""

    def __init__(self, detector: Any):
        if detector is None:
            raise ValueError("PacketCaptureService requires a detector implementation")
        self.sniffer = PacketSniffer(detector=detector)
        self._lock = asyncio.Lock()

    async def start_monitoring(self, interface: str = None):
        """Start monitoring on specified interface"""
        async with self._lock:
            if interface:
                self.sniffer.interface = interface
            await self.sniffer.start_capture()

    async def stop_monitoring(self):
        """Stop monitoring"""
        async with self._lock:
            await self.sniffer.stop_capture()

    def add_detection_callback(self, callback: Callable):
        """Add callback for detection events"""
        self.sniffer.add_event_callback(callback)

    async def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics"""
        return {
            'interface': self.sniffer.interface,
            'running': self.sniffer.running,
            'queue_size': self.sniffer.queue.queue.qsize(),
            'flows_count': len(self.sniffer.extractor.flows),
            'ip_count': len(self.sniffer.extractor.ip_stats)
        }