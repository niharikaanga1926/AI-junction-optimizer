import os
import sys
import json
import asyncio
import base64
from datetime import datetime
from typing import Optional
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse

from detection.detector import VehicleDetector
from traffic_signal.controller import SignalController
from ml.lstm_forecaster import CongestionForecaster, train_lstm

import features_engine
from features_router import router as features_router


# ─────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Junction Optimizer",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(features_router)


# ─────────────────────────────────────────────────────────────
# Global Variables
# ─────────────────────────────────────────────────────────────

ARM_NAMES = ["North", "South", "East", "West"]

main_loop = None

# FIX: Dedicated snapshot counter so forecast trigger works correctly.
# Using len(density_history) % 10 was unreliable once the deque hit maxlen=300
# because len stays at 300 and the modulo could re-fire on the same index.
_snapshot_count = 0


# ─────────────────────────────────────────────────────────────
# WebSocket Manager
# ─────────────────────────────────────────────────────────────

class ConnectionManager:
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


signal_manager = ConnectionManager()
density_manager = ConnectionManager()
frame_manager = ConnectionManager()
forecast_manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────
# Core Services
# ─────────────────────────────────────────────────────────────

VIDEO_SOURCE = "videos/traffic.mp4"

detector = VehicleDetector(source=VIDEO_SOURCE, model_path="yolov8n.pt", confidence=0.25)
controller = SignalController()
forecaster = CongestionForecaster("models/lstm_forecaster.pt")

density_history = deque(maxlen=300)
session_log = []

CO2_PER_VEHICLE_G_PER_MIN = 14.0


# ─────────────────────────────────────────────────────────────
# Startup Event
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():

    global main_loop, _snapshot_count
    main_loop = asyncio.get_running_loop()

    forecaster.load()

    # ---------------------------------------------------------
    # Detection Callback
    # ---------------------------------------------------------

    async def on_detection(snapshot: dict):

        global _snapshot_count

        arms = snapshot.get("arms", {})
        emergency_arm = None

        for arm, info in arms.items():

            score = info.get("density_score", 0)

            controller.update_density(arm, score)

            if info.get("emergency_detected"):
                emergency_arm = arm

        if emergency_arm:
            controller.trigger_emergency(emergency_arm)

        density_history.append(snapshot)
        session_log.append(snapshot)

        _snapshot_count += 1

        await density_manager.broadcast({
            "type": "density",
            "data": snapshot
        })

        # FIX: Use dedicated counter instead of len(density_history) % 10
        # len() stays at 300 once the deque is full, causing incorrect triggers
        if _snapshot_count % 10 == 0:

            logs = list(density_history)

            fcast = forecaster.predict(logs)

            await forecast_manager.broadcast({
                "type": "forecast",
                "data": fcast
            })

    detector.output_callback = on_detection

    # ---------------------------------------------------------
    # Signal Change Callback
    # ---------------------------------------------------------

    def on_signal_change(state: dict):

        if main_loop:
            asyncio.run_coroutine_threadsafe(
                signal_manager.broadcast({
                    "type": "signal",
                    "data": state
                }),
                main_loop
            )

    controller.on_state_change = on_signal_change

    # ---------------------------------------------------------
    # Start Services
    # ---------------------------------------------------------

    detector.start()

    # FIX: Inject the FastAPI event loop into the detector so its
    # background thread can post coroutines safely via run_coroutine_threadsafe.
    # Without this, the detector used asyncio.run() which creates a NEW loop
    # per call and silently drops all snapshots.
    detector._main_loop = main_loop

    controller.start()

    features_engine.init_engine(controller, density_history, session_log)

    asyncio.create_task(_frame_broadcast_loop())

    print("[API] Startup completed successfully.")


# ─────────────────────────────────────────────────────────────
# Shutdown Event
# ─────────────────────────────────────────────────────────────

@app.on_event("shutdown")
async def shutdown():

    try:
        detector.stop()
    except:
        pass

    try:
        controller.stop()
    except:
        pass

    print("[API] Shutdown complete.")


# ─────────────────────────────────────────────────────────────
# Frame Broadcasting
# ─────────────────────────────────────────────────────────────

async def _frame_broadcast_loop():

    while True:

        await asyncio.sleep(0.2)

        frame_bytes = detector.latest_frame_bytes

        if frame_bytes and frame_manager.active:

            b64 = base64.b64encode(frame_bytes).decode()

            await frame_manager.broadcast({
                "type": "frame",
                "data": b64
            })


# ─────────────────────────────────────────────────────────────
# WebSocket Routes
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws/signals")
async def ws_signals(ws: WebSocket):

    await signal_manager.connect(ws)

    await ws.send_json({
        "type": "signal",
        "data": controller.get_state()
    })

    try:
        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        signal_manager.disconnect(ws)


@app.websocket("/ws/density")
async def ws_density(ws: WebSocket):

    await density_manager.connect(ws)

    if density_history:
        await ws.send_json({
            "type": "density",
            "data": density_history[-1]
        })

    try:
        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        density_manager.disconnect(ws)


@app.websocket("/ws/frame")
async def ws_frame(ws: WebSocket):

    await frame_manager.connect(ws)

    try:
        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        frame_manager.disconnect(ws)


@app.websocket("/ws/forecast")
async def ws_forecast(ws: WebSocket):

    await forecast_manager.connect(ws)

    # FIX: Always send an initial forecast on connect.
    # Previously, if density_history was non-empty but had <5 snapshots,
    # nothing was sent and the frontend showed stale 0% bars forever.
    if len(density_history) >= 5:
        fcast = forecaster.predict(list(density_history))
    else:
        # Send a zeroed forecast so the frontend renders cleanly
        fcast = {
            "arms": {arm: {"density_forecast": [0] * 5} for arm in ARM_NAMES}
        }

    await ws.send_json({
        "type": "forecast",
        "data": fcast
    })

    try:
        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        forecast_manager.disconnect(ws)


# ─────────────────────────────────────────────────────────────
# REST APIs
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.get("/api/status")
def status():

    state = controller.get_state()

    density = {
        arm: density_history[-1]["arms"].get(arm, {})
        if density_history else {}
        for arm in ARM_NAMES
    }

    return {
        "status": "ok",
        "signals": state,
        "density": density
    }


@app.get("/api/kpis")
def kpis():

    if not density_history:
        return {
            "avg_wait_time": 0,
            "throughput": 0,
            "emergency_events": 0,
            "co2_g_per_min": 0
        }

    recent = list(density_history)[-30:]

    avg_density = np.mean([
        sum(
            s.get("arms", {}).get(a, {}).get("density_score", 0)
            for a in ARM_NAMES
        )
        for s in recent
    ]) if recent else 0

    total_vehicles = sum(
        sum(
            s.get("arms", {}).get(a, {}).get("vehicle_count", 0)
            for a in ARM_NAMES
        )
        for s in recent
    )

    throughput = round(
        total_vehicles / max(1, len(recent) * 2 / 60),
        1
    )

    emergency_events = sum(
        1 for inc in controller._incidents
        if "Emergency" in inc.get("message", "")
    )

    last = density_history[-1] if density_history else {}

    idle_vehicles = sum(
        last.get("arms", {}).get(a, {}).get("vehicle_count", 0)
        for a in ARM_NAMES
        if controller.signals.get(a, None)
        and controller.signals[a].value != "green"
    )

    co2 = round(
        idle_vehicles * CO2_PER_VEHICLE_G_PER_MIN,
        1
    )

    return {
        "avg_wait_time": round(avg_density * 0.9, 1),
        "throughput": throughput,
        "emergency_events": emergency_events,
        "co2_g_per_min": co2,
    }


@app.get("/api/history")
def history(limit: int = 100):
    return {"history": list(density_history)[-limit:]}


@app.get("/api/incidents")
def incidents():
    return {
        "incidents":
        controller._incidents[-50:] +
        detector.incidents[-50:]
    }


@app.get("/api/lane_suggestions")
def lane_suggestions():
    return {
        "suggestions":
        controller.lane_reversal_suggestions[-10:]
    }


@app.get("/api/forecast")
def forecast():

    if not density_history:
        return {"error": "No data yet"}

    return forecaster.predict(list(density_history))


# ─────────────────────────────────────────────────────────────
# Emergency APIs
# ─────────────────────────────────────────────────────────────

class EmergencyRequest(BaseModel):
    arm: str


@app.post("/api/emergency/trigger")
def trigger_emergency(req: EmergencyRequest):

    if req.arm not in ARM_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid arm. Choose from {ARM_NAMES}"
        )

    controller.trigger_emergency(req.arm)

    return {
        "status": "emergency_triggered",
        "arm": req.arm
    }


@app.post("/api/emergency/clear")
def clear_emergency():

    controller.clear_emergency()

    return {
        "status": "emergency_cleared"
    }


# ─────────────────────────────────────────────────────────────
# ML Training
# ─────────────────────────────────────────────────────────────

@app.post("/api/train/lstm")
async def train_model(background_tasks: BackgroundTasks):

    def _train():

        logs = list(density_history)

        train_lstm(
            logs=logs if len(logs) > 50 else None,
            epochs=20,
            save_path="models/lstm_forecaster.pt"
        )

        forecaster.load()

    background_tasks.add_task(_train)

    return {"status": "training_started"}


# ─────────────────────────────────────────────────────────────
# Session Replay
# ─────────────────────────────────────────────────────────────

@app.get("/api/session/replay")
def session_replay():
    return {"session": session_log}


# ─────────────────────────────────────────────────────────────
# Main Entrypoint
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )