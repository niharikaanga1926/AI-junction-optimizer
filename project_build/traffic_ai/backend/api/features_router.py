"""
features_router.py — AI Junction Optimizer: New Feature Endpoints
Mount this router in main.py (see instructions at bottom of file).
All reads go through features_engine — never duplicates existing state.
"""

import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

import features_engine as engine

# ─────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="", tags=["features"])


# ─────────────────────────────────────────────────────────────
# WebSocket Manager (features-specific)
# ─────────────────────────────────────────────────────────────

class FeatureConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


features_manager = FeatureConnectionManager()


# ─────────────────────────────────────────────────────────────
# REST Endpoints
# ─────────────────────────────────────────────────────────────

class ApiKeyRequest(BaseModel):
    api_key: str


@router.post("/api/features/apikey")
async def set_api_key(req: ApiKeyRequest):
    if not req.api_key or not req.api_key.startswith("sk-"):
        raise HTTPException(status_code=400, detail="Invalid API key format. Must start with 'sk-'.")
    engine.ANTHROPIC_API_KEY = req.api_key.strip()
    return {"status": "ok", "message": "API key accepted. AI briefings enabled."}


class SimulateRequest(BaseModel):
    scenario: str


@router.post("/api/features/simulate/start")
async def start_simulation(req: SimulateRequest):
    try:
        state = engine.start_simulation(req.scenario)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"status": "simulation_started", "simulation": state}


@router.post("/api/features/simulate/stop")
async def stop_simulation():
    state = engine.stop_simulation()
    return {"status": "simulation_stopped", "simulation": state}


@router.get("/api/features/lane_suggestions")
async def get_lane_suggestions():
    suggestions = engine.compute_lane_suggestions()
    return {"suggestions": suggestions}


@router.get("/api/features/efficiency")
async def get_efficiency():
    return engine.compute_efficiency()


@router.get("/api/features/carbon")
async def get_carbon():
    return engine.get_carbon_snapshot()


@router.get("/api/features/briefing")
async def get_briefing():
    briefing = await engine.refresh_ai_briefing()
    return briefing


@router.get("/api/features/simulation")
async def get_simulation():
    return engine.simulation


@router.get("/api/features/incidents")
async def get_sim_incidents():
    return {"incidents": engine.get_sim_incidents()}


@router.get("/api/features/status")
async def get_status():
    return {"status": "ok", "features": "connected"}


# ─────────────────────────────────────────────────────────────
# WebSocket /ws/features
# ─────────────────────────────────────────────────────────────

@router.websocket("/ws/features")
async def ws_features(ws: WebSocket):
    await features_manager.connect(ws)
    try:
        while True:
            await asyncio.sleep(1.0)

            engine.tick_simulation()
            engine.tick_carbon()

            lane_suggestions = engine.compute_lane_suggestions()
            briefing = await engine.refresh_ai_briefing()
            efficiency = engine.compute_efficiency()
            sim_state = engine.simulation
            carbon = engine.get_carbon_snapshot()

            payload = {
                "type": "features_update",
                "timestamp": datetime.utcnow().isoformat(),
                "lane_suggestions": lane_suggestions[-5:],
                "ai_briefing": briefing,
                "efficiency": efficiency,
                "simulation": {
                    "active": sim_state["active"],
                    "scenario": sim_state.get("scenario"),
                    "label": sim_state.get("label"),
                    "ends_in": (
                        max(0, int(sim_state["ends_at"] - __import__("time").time()))
                        if sim_state["active"] and sim_state.get("ends_at")
                        else 0
                    ),
                },
                "carbon": carbon,
            }

            await ws.send_json(payload)

    except WebSocketDisconnect:
        features_manager.disconnect(ws)
    except Exception as exc:
        print(f"[features WS] error: {exc}")
        features_manager.disconnect(ws)