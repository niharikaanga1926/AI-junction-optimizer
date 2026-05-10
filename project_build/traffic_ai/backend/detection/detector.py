"""
Step 1: Perception Engine
Detects vehicles per lane, classifies types, computes density score (0-100)
per junction arm every 2 seconds.

FIXES:
  - asyncio.run() replaced with run_coroutine_threadsafe() for thread safety
  - ARM_ROIS expanded slightly to eliminate center dead-zone
  - _main_loop injected from main.py after detector.start()
"""

import cv2
import json
import time
import math
import asyncio
import threading
from datetime import datetime
from typing import Optional
from collections import deque

import numpy as np
from ultralytics import YOLO

# ── Constants ────────────────────────────────────────────────────────────────

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

COCO_PERSON_CLASS = 0

ARM_NAMES = ["North", "South", "East", "West"]

# FIX: Expanded ROIs slightly so vehicles near the junction center
# are not silently dropped (old zones had a gap at cx/cy == 0.5)
ARM_ROIS = {
    "North": (0.0,  0.0,  0.55, 0.55),
    "South": (0.45, 0.45, 1.0,  1.0),
    "East":  (0.45, 0.0,  1.0,  0.55),
    "West":  (0.0,  0.45, 0.55, 1.0),
}

MAX_VEHICLES_FOR_MAX_DENSITY = 20

TRACK_HISTORY_LEN = 15

# ── Resize target — change here if needed ────────────────────────────────────
PROCESS_WIDTH  = 1280
PROCESS_HEIGHT = 720


class VehicleDetector:

    def __init__(
        self,
        source: str = "0",
        model_path: str = "yolov8n.pt",
        output_callback=None,
        confidence: float = 0.4,
    ):

        self.source = source
        self.model_path = model_path
        self.confidence = confidence
        self.output_callback = output_callback

        self.model: Optional[YOLO] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self.running = False

        self.latest_frame_bytes = None
        self.latest_snapshot = {}

        self.track_history = {}

        self.density_log = []

        self.incidents = []

        self._wrong_way_cache = {}

        self._last_push = time.time()

        # FIX: Will be set by main.py after detector.start()
        # so the detection thread can safely post coroutines
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None

    # ──────────────────────────────────────────────────────────────────────

    def load_model(self):
        print(f"[Detector] Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        print("[Detector] Model loaded.")

    def open_source(self):

        try:
            src = int(self.source)
        except ValueError:
            src = self.source

        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")

        print(f"[Detector] Opened source: {self.source}")

    # ──────────────────────────────────────────────────────────────────────

    def start(self):

        self.load_model()
        self.open_source()

        self.running = True

        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

        print("[Detector] Detection loop started.")

        return t

    def stop(self):

        self.running = False

        if self.cap:
            self.cap.release()

    # ──────────────────────────────────────────────────────────────────────

    def _loop(self):

        frame_buffer = []

        interval = 2.0

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                # Rewind video file to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize 4K/large frame down before any processing
            frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

            h, w = frame.shape[:2]

            results = self.model.track(
                frame,
                persist=True,
                conf=self.confidence,
                verbose=False,
            )[0]

            annotated = self._annotate_frame(
                frame.copy(),
                results,
                w,
                h,
            )

            _, buf = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )

            self.latest_frame_bytes = buf.tobytes()

            frame_buffer.append((results, w, h))

            now = time.time()

            if now - self._last_push >= interval:

                snapshot = self._compute_snapshot(
                    frame_buffer,
                    w,
                    h,
                )

                self.latest_snapshot = snapshot

                self.density_log.append({
                    "ts": snapshot["timestamp"],
                    "arms": snapshot["arms"],
                })

                # FIX: Robust async dispatch from background thread.
                # Try _main_loop first (set by main.py after start()).
                # Fall back to get_event_loop() in case of timing issues.
                if self.output_callback:
                    loop = self._main_loop
                    if loop is None:
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = None

                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._safe_callback(snapshot), loop
                        )
                        # Debug: shows vehicle counts per arm in terminal
                        counts = {
                            a: snapshot["arms"][a]["vehicle_count"]
                            for a in snapshot["arms"]
                        }
                        print(f"[Detector] Snapshot dispatched — {counts}")
                    else:
                        print("[Detector] WARNING: No running event loop — snapshot dropped.")

                frame_buffer.clear()

                self._last_push = now

    async def _safe_callback(self, snapshot):

        try:
            await self.output_callback(snapshot)
        except Exception as e:
            print(f"[Detector] Callback error: {e}")

    # ──────────────────────────────────────────────────────────────────────

    def _compute_snapshot(self, frame_buffer, w, h):

        arm_counts = {
            arm: {
                "car": 0,
                "truck": 0,
                "bus": 0,
                "motorcycle": 0,
                "person": 0,
                "emergency": False,
            }
            for arm in ARM_NAMES
        }

        wrong_way_alerts = []

        for results, fw, fh in frame_buffer:

            if results.boxes is None:
                continue

            for box in results.boxes:

                cls_id = int(box.cls[0])

                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(float, box.xyxy[0])

                cx = (x1 + x2) / 2 / fw
                cy = (y1 + y2) / 2 / fh

                track_id = -1

                if box.id is not None:
                    track_id = int(box.id[0])

                arm = self._get_arm(cx, cy)

                if arm is None:
                    continue

                if cls_id in VEHICLE_CLASSES:

                    vtype = VEHICLE_CLASSES[cls_id]

                    arm_counts[arm][vtype] += 1

                    if cls_id == 7 and conf > 0.80:
                        arm_counts[arm]["emergency"] = True

                        self._log_incident(
                            "EMERGENCY_DETECTED",
                            arm,
                            "Emergency vehicle detected",
                        )

                elif cls_id == COCO_PERSON_CLASS:
                    arm_counts[arm]["person"] += 1

                if track_id != -1:

                    if track_id not in self.track_history:
                        self.track_history[track_id] = deque(
                            maxlen=TRACK_HISTORY_LEN
                        )

                    self.track_history[track_id].append((cx, cy))

                    wrong = self._check_wrong_way(
                        track_id,
                        arm,
                    )

                    if wrong:

                        wrong_way_alerts.append({
                            "track_id": track_id,
                            "arm": arm,
                        })

                        self._log_incident(
                            "WRONG_WAY",
                            arm,
                            f"Vehicle {track_id} moving wrong direction",
                        )

        arms_out = {}

        for arm, counts in arm_counts.items():

            total_vehicles = (
                counts["car"]
                + counts["truck"]
                + counts["bus"]
                + counts["motorcycle"]
            )

            density = self._compute_density(total_vehicles)

            arms_out[arm] = {
                "vehicle_count": total_vehicles,
                "breakdown": {
                    "car": counts["car"],
                    "truck": counts["truck"],
                    "bus": counts["bus"],
                    "motorcycle": counts["motorcycle"],
                    "pedestrian": counts["person"],
                },
                "density_score": density,
                "emergency_detected": counts["emergency"],
            }

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "arms": arms_out,
            "wrong_way_alerts": wrong_way_alerts,
            "incidents": self.incidents[-20:],
        }

    # ──────────────────────────────────────────────────────────────────────

    def _get_arm(self, cx, cy):

        for arm, (x1, y1, x2, y2) in ARM_ROIS.items():

            if x1 <= cx < x2 and y1 <= cy < y2:
                return arm

        return None

    # ──────────────────────────────────────────────────────────────────────

    def _compute_density(self, vehicle_count):

        return min(
            100.0,
            round(
                (vehicle_count / MAX_VEHICLES_FOR_MAX_DENSITY) * 100,
                1,
            ),
        )

    # ──────────────────────────────────────────────────────────────────────

    def _check_wrong_way(self, track_id, arm):

        hist = self.track_history.get(track_id)

        if not hist or len(hist) < 8:
            return False

        pts = list(hist)

        start_x, start_y = pts[0]
        end_x, end_y = pts[-1]

        dx = end_x - start_x
        dy = end_y - start_y

        movement = math.sqrt(dx * dx + dy * dy)

        if movement < 0.03:
            return False

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        wrong = False

        if arm in ("North", "South"):

            if abs_dx > abs_dy * 1.5:
                wrong = True

        elif arm in ("East", "West"):

            if abs_dy > abs_dx * 1.5:
                wrong = True

        now = time.time()

        last_alert = self._wrong_way_cache.get(track_id, 0)

        if wrong:

            if now - last_alert > 10:

                self._wrong_way_cache[track_id] = now

                return True

        return False

    # ──────────────────────────────────────────────────────────────────────

    def _log_incident(self, event_type, arm, action):

        self.incidents.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "arm": arm,
            "action": action,
        })

    # ──────────────────────────────────────────────────────────────────────

    def _annotate_frame(self, frame, results, w, h):

        colors_arm = {
            "North": (0, 200, 100),
            "South": (0, 100, 200),
            "East": (200, 150, 0),
            "West": (150, 0, 200),
        }

        for arm, (x1, y1, x2, y2) in ARM_ROIS.items():

            pt1 = (int(x1 * w), int(y1 * h))
            pt2 = (int(x2 * w), int(y2 * h))

            cv2.rectangle(frame, pt1, pt2, colors_arm[arm], 2)

            cv2.putText(
                frame,
                arm,
                (pt1[0] + 5, pt1[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colors_arm[arm],
                2,
            )

        if results.boxes is None:
            return frame

        for box in results.boxes:

            cls_id = int(box.cls[0])

            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls_id in VEHICLE_CLASSES:

                label = VEHICLE_CLASSES[cls_id]

                color = (0, 255, 0)

            elif cls_id == COCO_PERSON_CLASS:

                label = "person"

                color = (255, 165, 0)

            else:
                continue

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        cv2.putText(
            frame,
            ts,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

        return frame


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "0"

    async def async_cb(snapshot):

        print(json.dumps(snapshot, indent=2, default=str))

    detector = VehicleDetector(
        source=source,
        output_callback=async_cb,
    )

    # When running standalone, grab the running loop after starting
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    t = detector.start()
    detector._main_loop = loop  # inject loop for thread-safe callbacks

    try:
        loop.run_forever()

    except KeyboardInterrupt:

        detector.stop()

        print("\n[Detector] Stopped.")