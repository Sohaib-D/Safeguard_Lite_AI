"""
Unit tests for packet capture engine
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pathlib import Path

from backend.services.packet_capture import (
    PacketEvent, PacketQueue, FeatureExtractor,
    PacketSniffer, PacketCaptureService
)
from backend.services.detection_engine import DetectionResult, DetectionEngine
from backend.schemas.alert import AlertSeverity
from backend.services.alert_service import AlertService
from backend.services.rule_parser import RuleParser
from backend.services.websocket_manager import WebSocketManager


@pytest.fixture
def temp_db(tmp_path: Path):
    db_path = tmp_path / "test_packet_capture.db"
    return str(db_path)


@pytest.fixture
def detection_engine(temp_db):
    ws = WebSocketManager()
    alert_service = AlertService(temp_db, ws)
    rule_parser = RuleParser("./does_not_exist")
    return DetectionEngine(alert_service=alert_service, rule_parser=rule_parser)


class TestPacketQueue:
    """Test PacketQueue functionality"""

    @pytest.mark.asyncio
    async def test_put_get(self):
        queue = PacketQueue(max_size=10)
        packet = PacketEvent(
            timestamp=datetime.now(),
            src_ip="192.168.1.1",
            dst_ip="192.168.1.2",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            length=100
        )

        await queue.put(packet)
        retrieved = await queue.get()
        assert retrieved.src_ip == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_queue_full(self):
        queue = PacketQueue(max_size=2)
        packet = PacketEvent(
            timestamp=datetime.now(),
            src_ip="192.168.1.1",
            dst_ip="192.168.1.2",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            length=100
        )

        # Fill queue
        await queue.put(packet)
        await queue.put(packet)

        # Third put should drop oldest
        await queue.put(packet)
        assert queue.queue.qsize() == 2


class TestFeatureExtractor:
    """Test feature extraction"""

    def test_extract_features(self):
        extractor = FeatureExtractor()
        packet = PacketEvent(
            timestamp=datetime.now(),
            src_ip="192.168.1.1",
            dst_ip="192.168.1.2",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            length=100,
            flags="S",
            payload_size=50
        )

        features = extractor.extract_features(packet)

        assert features['src_ip'] == "192.168.1.1"
        assert features['protocol'] == "TCP"
        assert features['is_syn'] is True
        assert features['packet_length'] == 100

    def test_flow_tracking(self):
        extractor = FeatureExtractor()
        packet1 = PacketEvent(
            timestamp=datetime.now(),
            src_ip="192.168.1.1",
            dst_ip="192.168.1.2",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            length=100
        )
        packet2 = PacketEvent(
            timestamp=datetime.now(),
            src_ip="192.168.1.1",
            dst_ip="192.168.1.2",
            src_port=12345,
            dst_port=80,
            protocol="TCP",
            length=200
        )

        extractor.extract_features(packet1)
        extractor.extract_features(packet2)

        flow_key = ("192.168.1.1", "192.168.1.2", 12345, 80, "TCP")
        assert flow_key in extractor.flows
        assert extractor.flows[flow_key]['packet_count'] == 2
        assert extractor.flows[flow_key]['byte_count'] == 300


class TestDetectionEngine:
    """Test detection engine"""

    @pytest.mark.asyncio
    async def test_port_scan_detection(self, detection_engine):
        features = {
            'src_ip': '192.168.1.1',
            'unique_ports': 15  # Above threshold
        }

        results = await detection_engine.detect(features)
        assert len(results) >= 1
        # Find the port scan alert
        port_scan_alerts = [r for r in results if r.alert_type == 'rapid_port_scan']
        assert len(port_scan_alerts) == 1

    @pytest.mark.asyncio
    async def test_syn_flood_detection(self, detection_engine):
        features = {
            'src_ip': '192.168.1.1',
            'bytes_per_minute': 1_500_000  # Above bandwidth threshold
        }

        results = await detection_engine.detect(features)
        assert len(results) >= 1
        bandwidth_alerts = [r for r in results if r.alert_type == 'abnormal_bandwidth']
        assert len(bandwidth_alerts) == 1

    @pytest.mark.asyncio
    async def test_dns_tunneling_detection(self, detection_engine):
        features = {
            'src_ip': '192.168.1.1',
            'dns_query_count': 25,  # Above suspicious_dns threshold of 20
        }

        results = await detection_engine.detect(features)
        assert len(results) >= 1
        dns_alerts = [r for r in results if r.alert_type == 'suspicious_dns']
        assert len(dns_alerts) == 1


class TestPacketSniffer:
    """Test packet sniffer"""

    @patch('backend.services.packet_capture.AsyncSniffer')
    @pytest.mark.asyncio
    async def test_start_stop_capture(self, mock_sniffer_class):
        mock_sniffer = AsyncMock()
        mock_sniffer_class.return_value = mock_sniffer

        mock_detector = Mock()
        sniffer = PacketSniffer(detector=mock_detector)
        await sniffer.start_capture()
        assert sniffer.running is True

        await sniffer.stop_capture()
        assert sniffer.running is False
        mock_sniffer.stop.assert_called_once()

    def test_parse_packet_tcp(self):
        mock_detector = Mock()
        sniffer = PacketSniffer(detector=mock_detector)

        # Mock scapy packet
        from scapy.all import IP, TCP
        mock_packet = Mock()
        mock_ip = Mock()
        mock_ip.src = "192.168.1.1"
        mock_ip.dst = "192.168.1.2"
        mock_tcp = Mock()
        mock_tcp.sport = 12345
        mock_tcp.dport = 80
        mock_tcp.flags = "S"
        mock_packet.__getitem__ = Mock(side_effect=lambda x: mock_ip if x == IP else mock_tcp if x == TCP else Mock())
        mock_packet.__contains__ = Mock(side_effect=lambda x: x in [IP, TCP])
        mock_packet.__len__ = Mock(return_value=100)
        mock_packet.payload = Mock()
        mock_packet.payload.__len__ = Mock(return_value=50)

        event = sniffer._parse_packet(mock_packet)
        assert event.src_ip == "192.168.1.1"
        assert event.protocol == "TCP"
        assert event.flags == "S"


class TestPacketCaptureService:
    """Test high-level service"""

    @pytest.mark.asyncio
    async def test_start_monitoring(self):
        mock_detector = Mock()
        service = PacketCaptureService(detector=mock_detector)
        with patch.object(service.sniffer, 'start_capture', new_callable=AsyncMock):
            await service.start_monitoring("eth0")
            assert service.sniffer.interface == "eth0"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        mock_detector = Mock()
        service = PacketCaptureService(detector=mock_detector)
        stats = await service.get_stats()
        assert 'running' in stats
        assert 'queue_size' in stats


# Integration test
@pytest.mark.asyncio
async def test_full_pipeline(temp_db):
    """Test the full packet processing pipeline"""
    queue = PacketQueue()
    extractor = FeatureExtractor()
    ws = WebSocketManager()
    alert_service = AlertService(temp_db, ws)
    rule_parser = RuleParser("./does_not_exist")
    detector = DetectionEngine(alert_service=alert_service, rule_parser=rule_parser)

    # Mock packet
    packet = PacketEvent(
        timestamp=datetime.now(),
        src_ip="192.168.1.1",
        dst_ip="192.168.1.2",
        src_port=12345,
        dst_port=80,
        protocol="TCP",
        length=100,
        flags="S"
    )

    # Process
    await queue.put(packet)
    retrieved_packet = await queue.get()
    features = extractor.extract_features(retrieved_packet)
    detections = await detector.detect(features)

    assert features['is_syn'] is True
    # Should detect something based on features