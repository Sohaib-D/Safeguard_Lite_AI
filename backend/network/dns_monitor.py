import time
from collections import defaultdict
from typing import List


class DNSMonitor:
    """Monitors DNS requests to identify malicious domains or tunneling."""

    # A tiny example list of known bad patterns (in reality, this would be a large threat feed)
    SUSPICIOUS_TLDS = {".xyz", ".top", ".pw", ".cc"}
    
    def __init__(self, time_window: int = 60, rate_threshold: int = 50):
        self.time_window = time_window
        self.rate_threshold = rate_threshold
        self.dns_requests = defaultdict(list)

    def analyze_query(self, src_ip: str, domain: str) -> List[str]:
        """Analyze a DNS query and return any alerts."""
        alerts = []
        current_time = time.time()
        
        self._cleanup(current_time)
        self.dns_requests[src_ip].append(current_time)

        # 1. Suspicious TLD Check
        if any(domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS):
            alerts.append(
                f"Suspicious Domain Alert: Device {src_ip} requested a potentially "
                f"unsafe domain: {domain}."
            )

        # 2. DNS Tunneling/Spam Check (High rate of queries)
        if len(self.dns_requests[src_ip]) > self.rate_threshold:
            alerts.append(
                f"High DNS Rate Warning: Device {src_ip} made {len(self.dns_requests[src_ip])} "
                f"DNS queries in {self.time_window} seconds. This could indicate malware or DNS tunneling."
            )

        # 3. Long Domain Names (Often used for data exfiltration via DNS)
        if len(domain) > 60:
            alerts.append(
                f"Suspiciously Long Domain: Device {src_ip} requested {domain}. "
                "Hackers sometimes hide stolen data inside long domain names!"
            )

        return alerts

    def _cleanup(self, current_time: float):
        cutoff = current_time - self.time_window
        for ip in list(self.dns_requests.keys()):
            self.dns_requests[ip] = [t for t in self.dns_requests[ip] if t > cutoff]
            if not self.dns_requests[ip]:
                del self.dns_requests[ip]
