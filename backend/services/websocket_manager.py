from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from fastapi import WebSocket


@dataclass
class WebSocketConnection:
    websocket: WebSocket
    channels: Set[str] = field(default_factory=set)


class WebSocketManager:
    def __init__(self) -> None:
        self.active_connections: List[WebSocketConnection] = []
        self.lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channels: list[str] | None = None) -> None:
        await websocket.accept()
        if channels is None or not channels:
            channels = ["alerts", "traffic", "notifications", "logs"]
        connection = WebSocketConnection(websocket=websocket, channels=set(channels))
        async with self.lock:
            self.active_connections.append(connection)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self.lock:
            self.active_connections = [
                conn for conn in self.active_connections if conn.websocket is not websocket
            ]

    async def broadcast(self, event_type: str, payload: Dict[str, Any]) -> None:
        message = json.dumps({"type": event_type, "payload": payload})
        disconnected: list[WebSocketConnection] = []
        async with self.lock:
            for connection in self.active_connections:
                if "all" not in connection.channels and event_type not in connection.channels:
                    continue
                try:
                    await connection.websocket.send_text(message)
                except Exception:
                    disconnected.append(connection)
            for connection in disconnected:
                self.active_connections.remove(connection)
