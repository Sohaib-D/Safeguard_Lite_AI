import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ConnectionEvent:
    src_ip: str
    dst_ip: str
    dst_port: int
    protocol: str
    timestamp: float
    is_auth_failure: bool = False


class ConnectionTracker:
    """Tracks network connections to identify suspicious patterns like brute force."""

    def __init__(self, time_window: int = 60, brute_force_threshold: int = 5):
        self.time_window = time_window
        self.brute_force_threshold = brute_force_threshold
        # Tracks connection attempts: (src_ip, dst_port) -> list of timestamps
        self.connections: Dict[tuple, List[float]] = defaultdict(list)
        # Tracks failed auth attempts: (src_ip, dst_port) -> list of timestamps
        self.auth_failures: Dict[tuple, List[float]] = defaultdict(list)

    def track(self, event: ConnectionEvent) -> List[str]:
        """Track an event and return any alerts triggered."""
        alerts = []
        current_time = time.time()
        
        # Clean up old records
        self._cleanup(current_time)

        key = (event.src_ip, event.dst_port)
        self.connections[key].append(current_time)

        if event.is_auth_failure:
            self.auth_failures[key].append(current_time)
            
            # Check for brute force
            if len(self.auth_failures[key]) >= self.brute_force_threshold:
                alerts.append(
                    f"Brute Force Warning: IP {event.src_ip} has failed authentication "
                    f"{len(self.auth_failures[key])} times on port {event.dst_port} "
                    f"within {self.time_window} seconds!"
                )

        # Check for rapid connection spikes (potential DoS/Scan)
        if len(self.connections[key]) > (self.brute_force_threshold * 4):
            alerts.append(
                f"Traffic Spike Warning: IP {event.src_ip} is making an unusually high "
                f"number of connections to port {event.dst_port}."
            )

        return alerts

    def _cleanup(self, current_time: float):
        """Remove events outside the time window."""
        cutoff = current_time - self.time_window
        
        for key in list(self.connections.keys()):
            self.connections[key] = [t for t in self.connections[key] if t > cutoff]
            if not self.connections[key]:
                del self.connections[key]

        for key in list(self.auth_failures.keys()):
            self.auth_failures[key] = [t for t in self.auth_failures[key] if t > cutoff]
            if not self.auth_failures[key]:
                del self.auth_failures[key]
