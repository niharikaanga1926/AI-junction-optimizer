"""
Step 4: LSTM Congestion Forecaster
Trains on historical density logs. Predicts next 10 minutes (20 steps × 30s).
Triggers pre-alert when predicted density > 70.
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ARM_NAMES = ["North", "South", "East", "West"]
SEQ_LEN  = 20    # look-back window (20 × 30s = 10 min)
PRED_LEN = 20    # forecast horizon (20 × 30s = 10 min)
ALERT_THRESHOLD = 70.0


# ── Dataset ────────────────────────────────────────────────────────────────────

class DensityDataset(Dataset):
    """
    Converts a list of density-log entries into sliding window (X, Y) pairs.
    Entry format: {"ts": "...", "arms": {"North":50, "South":20, ...}}
    """

    def __init__(self, logs: list, seq_len: int = SEQ_LEN, pred_len: int = PRED_LEN):
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Build (n_steps, 4) array from logs
        series = []
        for entry in logs:
            arms = entry["arms"]
            if isinstance(arms, dict):
                # Support both flat {"North":50} and nested {"North":{"density_score":50}}
                row = []
                for arm in ARM_NAMES:
                    val = arms.get(arm, 0)
                    if isinstance(val, dict):
                        val = val.get("density_score", 0)
                    row.append(float(val))
                series.append(row)

        self.data = np.array(series, dtype=np.float32) / 100.0  # normalise

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]                      # (seq_len, 4)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]  # (pred_len, 4)
        return torch.tensor(x), torch.tensor(y)


# ── Model ──────────────────────────────────────────────────────────────────────

class TrafficLSTM(nn.Module):
    """
    Encoder-Decoder LSTM.
    Input:  (batch, seq_len, 4)
    Output: (batch, pred_len, 4)  — density predictions per arm
    """

    def __init__(self, input_size=4, hidden=64, n_layers=2, pred_len=PRED_LEN):
        super().__init__()
        self.pred_len = pred_len
        self.encoder = nn.LSTM(input_size, hidden, n_layers, batch_first=True, dropout=0.2)
        self.decoder = nn.LSTM(input_size, hidden, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, input_size)

    def forward(self, x, target=None, teacher_force_ratio=0.5):
        batch = x.size(0)
        _, (h, c) = self.encoder(x)

        # Initialise decoder input with last encoder input step
        dec_in = x[:, -1:, :]   # (batch, 1, 4)
        outputs = []

        for t in range(self.pred_len):
            out, (h, c) = self.decoder(dec_in, (h, c))
            pred = self.fc(out)              # (batch, 1, 4)
            outputs.append(pred)
            # Teacher forcing during training
            if target is not None and torch.rand(1).item() < teacher_force_ratio:
                dec_in = target[:, t:t+1, :]
            else:
                dec_in = pred.detach()

        return torch.cat(outputs, dim=1)    # (batch, pred_len, 4)


# ── Training ───────────────────────────────────────────────────────────────────

def generate_synthetic_logs(n: int = 2000) -> list:
    """Generate synthetic density logs for training when no real data exists."""
    logs = []
    densities = {arm: np.random.uniform(10, 60) for arm in ARM_NAMES}
    for i in range(n):
        for arm in ARM_NAMES:
            # Simulate rush-hour spikes
            t = i / n
            rush = 30 * np.sin(2 * np.pi * t * 3) ** 2
            noise = np.random.normal(0, 5)
            densities[arm] = np.clip(densities[arm] + noise + rush * 0.1, 0, 100)
        logs.append({
            "ts": datetime.utcnow().isoformat(),
            "arms": {arm: round(densities[arm], 1) for arm in ARM_NAMES}
        })
    return logs


def train_lstm(
    logs: Optional[list] = None,
    epochs: int = 30,
    batch_size: int = 32,
    save_path: str = "models/lstm_forecaster.pt",
):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if logs is None or len(logs) < SEQ_LEN + PRED_LEN + 10:
        print("[LSTM] Not enough logs — using synthetic data for training.")
        logs = generate_synthetic_logs(3000)

    dataset = DensityDataset(logs)
    if len(dataset) == 0:
        print("[LSTM] Dataset too small, skipping training.")
        return None

    split = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(dataset, [split, len(dataset) - split])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficLSTM().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x, target=y)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        avg_train = train_loss / max(1, len(train_loader))
        avg_val   = val_loss   / max(1, len(val_loader))
        print(f"[LSTM] Epoch {epoch:3d}/{epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), save_path)

    print(f"[LSTM] Best model saved → {save_path}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

class CongestionForecaster:
    """
    Wraps trained LSTM for real-time inference.
    Feed density logs; get 10-min forecast + alert flag.
    """

    def __init__(self, model_path: str = "models/lstm_forecaster.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrafficLSTM().to(self.device)
        self.model_path = model_path
        self._loaded = False

    def load(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self._loaded = True
            print(f"[LSTM] Loaded model from {self.model_path}")
        else:
            print(f"[LSTM] No model found at {self.model_path} — run training first.")

    def predict(self, recent_logs: list) -> dict:
        """
        recent_logs: last SEQ_LEN density-log entries
        Returns forecast dict with per-arm predictions and alert flag.
        """
        if not self._loaded:
            # Return dummy forecast if model not ready
            return self._dummy_forecast(recent_logs)

        ds = DensityDataset(recent_logs, seq_len=SEQ_LEN, pred_len=PRED_LEN)
        if len(ds) == 0:
            return self._dummy_forecast(recent_logs)

        x, _ = ds[len(ds) - 1]
        x = x.unsqueeze(0).to(self.device)   # (1, seq_len, 4)

        with torch.no_grad():
            pred = self.model(x)              # (1, pred_len, 4)

        pred_np = pred.squeeze(0).cpu().numpy() * 100.0  # denorm
        pred_np = np.clip(pred_np, 0, 100)

        # Per-arm forecast series
        forecast = {}
        alert = False
        for i, arm in enumerate(ARM_NAMES):
            series = pred_np[:, i].tolist()
            max_pred = float(max(series))
            forecast[arm] = {
                "series": [round(v, 1) for v in series],
                "max_predicted": round(max_pred, 1),
                "alert": max_pred > ALERT_THRESHOLD,
            }
            if max_pred > ALERT_THRESHOLD:
                alert = True

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "forecast_horizon_minutes": round(PRED_LEN * 0.5, 1),
            "arms": forecast,
            "global_alert": alert,
        }

    def _dummy_forecast(self, logs: list) -> dict:
        """Fallback when model not trained — returns last-known + small noise."""
        last = logs[-1]["arms"] if logs else {}
        forecast = {}
        for arm in ARM_NAMES:
            base = last.get(arm, 0)
            if isinstance(base, dict):
                base = base.get("density_score", 0)
            series = [round(float(base) + np.random.normal(0, 3), 1) for _ in range(PRED_LEN)]
            series = [max(0, min(100, v)) for v in series]
            forecast[arm] = {"series": series, "max_predicted": max(series), "alert": max(series) > ALERT_THRESHOLD}
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "forecast_horizon_minutes": round(PRED_LEN * 0.5, 1),
            "arms": forecast,
            "global_alert": any(v["alert"] for v in forecast.values()),
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "train":
        train_lstm(epochs=20, save_path="models/lstm_forecaster.pt")

    elif mode == "predict":
        logs = generate_synthetic_logs(50)
        fc = CongestionForecaster("models/lstm_forecaster.pt")
        fc.load()
        result = fc.predict(logs)
        print(json.dumps(result, indent=2))
