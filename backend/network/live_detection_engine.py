from typing import List, Callable, Any
from backend.network.connection_tracker import ConnectionTracker, ConnectionEvent
from backend.network.dns_monitor import DNSMonitor
from backend.network.traffic_analyzer import TrafficAnalyzer


class LiveDetectionEngine:
    """Real-time engine that processes packet events through various analyzers."""

    def __init__(self):
        self.conn_tracker = ConnectionTracker()
        self.dns_monitor = DNSMonitor()
        self.traffic_analyzer = TrafficAnalyzer()
        self.callbacks: List[Callable] = []

    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)

    async def _emit_alert(self, alert_message: str):
        for callback in self.callbacks:
            try:
                await callback({
                    "type": "live_alert",
                    "description": alert_message,
                    "severity": "High"
                })
            except Exception:
                pass

    async def process_packet(self, packet_data: dict):
        """Process an incoming packet dictionary from the sniffer."""
        alerts = []

        src_ip = packet_data.get("src_ip")
        dst_ip = packet_data.get("dst_ip")
        protocol = packet_data.get("protocol")
        length = packet_data.get("length", 0)

        if not src_ip or not dst_ip:
            return

        # Traffic Volume Check
        alerts.extend(self.traffic_analyzer.analyze(src_ip, dst_ip, length))

        # Connection / Port Tracking
        if protocol in ("TCP", "UDP"):
            dst_port = packet_data.get("dst_port")
            if dst_port:
                is_auth_failure = False
                # Simple heuristic: RST packets on SSH/FTP might imply failed auth
                if packet_data.get("flags") and 'R' in packet_data["flags"]:
                    if dst_port in (21, 22, 3389):
                        is_auth_failure = True
                
                event = ConnectionEvent(
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    dst_port=dst_port,
                    protocol=protocol,
                    timestamp=packet_data.get("timestamp_val", 0.0),
                    is_auth_failure=is_auth_failure
                )
                alerts.extend(self.conn_tracker.track(event))

        # DNS Checking
        if protocol == "UDP" and packet_data.get("dst_port") == 53:
            domain = packet_data.get("dns_query")
            if domain:
                alerts.extend(self.dns_monitor.analyze_query(src_ip, domain))

        for alert in alerts:
            await self._emit_alert(alert)
