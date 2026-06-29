import sys
from pathlib import Path
from typing import Dict


def ensure_virtual_environment() -> None:
    if sys.prefix == sys.base_prefix:
        raise SystemExit(
            "This app must run inside a virtual environment. "
            "Use .\\run.ps1 from the project root."
        )


ensure_virtual_environment()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent


@app.get("/")
async def serve_frontend():
    return FileResponse(BASE_DIR / "index.html")


# Map to store connected users: { "username": WebSocket }
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: dict, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(user_id, websocket)
    try:
        while True:
            # Receive the signaling data from the frontend
            data = await websocket.receive_json()

            target_user = data.get("target")
            # Expected types: 'offer', 'answer', or 'candidate'.
            message_type = data.get("type")
            payload = data.get("payload")

            # Route the message to the specific target user
            await manager.send_personal_message(
                {"from": user_id, "type": message_type, "payload": payload},
                target_user,
            )

    except WebSocketDisconnect:
        manager.disconnect(user_id)
