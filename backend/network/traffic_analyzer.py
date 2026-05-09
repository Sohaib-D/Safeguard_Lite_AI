from collections import defaultdict
import time
from typing import List

class TrafficAnalyzer:
    """Analyzes traffic volume to detect DoS or data exfiltration."""

    def __init__(self, time_window: int = 60, max_bytes_per_window: int = 50_000_000):
        # Default: warn if an IP sends/receives > 50MB in 60 seconds
        self.time_window = time_window
        self.max_bytes = max_bytes_per_window
        self.ip_bytes_tracker = defaultdict(int)
        self.window_start = time.time()

    def analyze(self, src_ip: str, dst_ip: str, length: int) -> List[str]:
        alerts = []
        current_time = time.time()

        # Reset window if needed
        if current_time - self.window_start > self.time_window:
            self.ip_bytes_tracker.clear()
            self.window_start = current_time

        self.ip_bytes_tracker[src_ip] += length
        self.ip_bytes_tracker[dst_ip] += length

        # Check for abnormal volume (Data Exfiltration or DoS)
        if self.ip_bytes_tracker[src_ip] > self.max_bytes:
            alerts.append(
                f"High Data Volume Warning: Device {src_ip} has transferred a massive "
                f"amount of data (>{self.max_bytes // 1_000_000} MB) very quickly. "
                "This could be a Denial of Service attack or someone stealing large files."
            )
            # Temporarily prevent spamming the same alert
            self.ip_bytes_tracker[src_ip] = -(self.max_bytes * 10)

        return alerts
