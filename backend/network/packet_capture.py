import asyncio
import logging
import time
from typing import Callable, List, Optional
from scapy.all import AsyncSniffer, IP, TCP, UDP, DNSQR

from backend.network.live_detection_engine import LiveDetectionEngine

logger = logging.getLogger(__name__)

class LivePacketCapture:
    """Captures live packets and feeds them into the LiveDetectionEngine."""

    def __init__(self, interface: str = None):
        self.interface = interface
        self.running = False
        self.sniffer: Optional[AsyncSniffer] = None
        self.engine = LiveDetectionEngine()
        self.loop = asyncio.get_event_loop()

    def add_callback(self, callback: Callable):
        self.engine.add_callback(callback)

    def _packet_handler(self, packet):
        """Callback for scapy when a packet arrives."""
        if not self.running:
            return

        try:
            if IP in packet:
                data = {
                    "src_ip": packet[IP].src,
                    "dst_ip": packet[IP].dst,
                    "length": len(packet),
                    "timestamp_val": time.time(),
                }
                if TCP in packet:
                    data["protocol"] = "TCP"
                    data["src_port"] = packet[TCP].sport
                    data["dst_port"] = packet[TCP].dport
                    data["flags"] = str(packet[TCP].flags)
                elif UDP in packet:
                    data["protocol"] = "UDP"
                    data["src_port"] = packet[UDP].sport
                    data["dst_port"] = packet[UDP].dport
                    if DNSQR in packet:
                        data["dns_query"] = packet[DNSQR].qname.decode('utf-8', errors='ignore')
                else:
                    data["protocol"] = "OTHER"

                # Schedule the async processing in the main event loop
                asyncio.run_coroutine_threadsafe(self.engine.process_packet(data), self.loop)
        except Exception as e:
            logger.error(f"Packet parsing error: {e}")

    async def start(self):
        if self.running:
            return
        self.running = True
        logger.info(f"Starting live packet capture on interface {self.interface or 'default'}")
        self.sniffer = AsyncSniffer(
            iface=self.interface,
            prn=self._packet_handler,
            store=0,
            filter="ip"
        )
        await self.sniffer.start()

    async def stop(self):
        self.running = False
        if self.sniffer:
            await self.sniffer.stop()
        logger.info("Live packet capture stopped")
