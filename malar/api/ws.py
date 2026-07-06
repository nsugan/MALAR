"""WebSocket event stream — stage outputs + agent events.

The UI subscribes to /ws and receives stage events (perception ... memory_write),
coverage updates, OOD flags and HITL queue changes. In review mode each stage event
carries the output the operator approves/edits/rejects.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


class WSManager:
    def __init__(self):
        self.active: list[WebSocket] = []
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def emit_threadsafe(self, message: dict):
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)


manager = WSManager()


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    manager.loop = asyncio.get_event_loop()
    await manager.connect(ws)
    try:
        await ws.send_text(json.dumps({"type": "hello", "msg": "MALAR event stream"}))
        while True:
            await ws.receive_text()  # keep-alive / client pings
    except WebSocketDisconnect:
        manager.disconnect(ws)
