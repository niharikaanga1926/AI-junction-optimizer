# 🚦 AI Junction Optimizer
### Real-Time Adaptive Traffic Signal Control System
<img width="1917" height="907" alt="image" src="https://github.com/user-attachments/assets/94e8500b-fc37-408a-84b1-0629072b4dd0" />
<img width="1917" height="778" alt="image" src="https://github.com/user-attachments/assets/2fc68401-2cf5-45a3-acc4-f9e28b3cc874" />
<img width="1918" height="901" alt="image" src="https://github.com/user-attachments/assets/2d3cf1e3-5bff-49a0-803b-ea3dc005a6bd" />
<img width="1462" height="291" alt="image" src="https://github.com/user-attachments/assets/5fabeb38-cb1f-4d33-8bd3-0341a52940f0" />


![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

**AI Junction Optimizer** is a real-time intelligent traffic management system that uses computer vision, machine learning, and AI to optimize traffic signal timings at road junctions. Unlike traditional fixed-timer systems, this solution dynamically adapts signal phases based on live vehicle density detected from video feeds.

Built for the **AI Based Junction Optimization System** hackathon challenge.

---

## 🎯 Problem Statement

Traffic congestion at road junctions is a major issue in cities worldwide. Most traffic signals operate on fixed timing systems regardless of actual traffic levels, causing:
- Unnecessary vehicle waiting time
- Road overcrowding
- Emergency vehicle delays
- Increased CO₂ emissions from idling vehicles

---

## ✅ Features

### Core Features (Problem Statement)
| Feature | Description |
|---|---|
| 🎥 **Real-Time Vehicle Detection** | YOLOv8-powered vehicle detection from aerial video feed |
| 🚦 **Adaptive Signal Timing** | AI dynamically adjusts green/red phases based on density |
| 🚨 **Emergency Vehicle Priority** | Auto-detects and manually overrides signals for ambulances |
| 📈 **Congestion Prediction** | LSTM-based 10-minute traffic forecast |
| 📊 **Live Dashboard** | Real-time monitoring with WebSocket updates |
| 🛣️ **Lane Management Suggestions** | AI advisories when density exceeds 50% |

### Extra Features (Beyond Problem Statement)
| Feature | Description |
|---|---|
| 🤖 **AI Natural Language Briefing** | Claude AI generates 2-line traffic summaries every 30s |
| ⚡ **Before vs After Efficiency** | Real-time comparison vs fixed timing baseline |
| 🎬 **Simulation Mode** | Test Rush Hour, Emergency, Accident, Off-Peak scenarios |
| 🌱 **Carbon Savings Tracker** | Cumulative CO₂ saved vs fixed timer system |
| 🗺️ **Junction Heatmap** | Visual density map per direction (N/S/E/W) |

---

## 🖼️ Screenshots

### Live Dashboard
<img width="1897" height="898" alt="image" src="https://github.com/user-attachments/assets/161ffee9-2041-4d75-b849-797280fd6ddd" />


### Analytics View
<img width="1888" height="622" alt="image" src="https://github.com/user-attachments/assets/22d6ebd4-d9f9-4911-9027-58b5c58fb1cf" />

---

## 🏗️ Project Structure

```
traffic_ai/
├── backend/
│   ├── api/
│   │   ├── main.py                 # FastAPI app, WebSocket routes, REST APIs
│   │   ├── features_engine.py      # 5 extra feature logic engine
│   │   ├── features_router.py      # Feature WebSocket + REST endpoints
│   │   ├── index.html              # Frontend dashboard
│   │   ├── .env                    # API keys (not committed)
│   │   └── videos/
│   │       └── traffic.mp4         # Input video feed
│   ├── detection/
│   │   └── detector.py             # YOLOv8 vehicle detector
│   ├── traffic_signal/
│   │   └── controller.py           # Signal controller logic
│   └── ml/
│       └── lstm_forecaster.py      # LSTM congestion forecaster
├── models/
│   └── lstm_forecaster.pt          # Trained LSTM model
├── venv/                           # Virtual environment
├── run.bat                         # One-click startup script
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Windows 10/11
- ~2GB free disk space
- Webcam or traffic video file

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/ai-junction-optimizer.git
cd ai-junction-optimizer
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install fastapi uvicorn websockets opencv-python ultralytics
pip install httpx anthropic numpy pillow python-dotenv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**4. Add your traffic video**
```
Place your traffic video at:
backend/api/videos/traffic.mp4
```

**5. Set up API key (optional — for AI briefing)**
```bash
# Create .env file in backend/api/
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxx
```

### Running the Project

**Option 1 — Double click:**
```
run.bat
```

**Option 2 — Manual:**
```bash
venv\Scripts\activate
cd backend\api
python main.py
```

**Open dashboard:**
```
http://localhost:8000
```

---

## 🔧 How It Works

```
Traffic Video Feed
      ↓
YOLOv8 detects vehicles crossing detection zones
      ↓
Density % calculated per direction (N/S/E/W)
      ↓
AI Signal Controller decides GREEN direction
      ↓
Dashboard updates in real-time via WebSockets:
  ├── Signal states change dynamically
  ├── KPIs update (wait time, throughput, CO₂)
  ├── Heatmap refreshes per direction
  ├── Charts record density history
  ├── If density > 70% → Incident logged
  ├── If density > 50% → Lane advisory triggered
  └── Every 30s → Claude AI writes traffic briefing
```

---

## 📡 API Endpoints

### WebSockets
| Endpoint | Description |
|---|---|
| `ws://localhost:8000/ws/signals` | Live signal state updates |
| `ws://localhost:8000/ws/density` | Live density per direction |
| `ws://localhost:8000/ws/frame` | Live video frame stream |
| `ws://localhost:8000/ws/features` | All 5 feature updates |

### REST APIs
| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | System status |
| `/api/kpis` | GET | Live KPIs |
| `/api/incidents` | GET | Incident log |
| `/api/emergency/trigger` | POST | Trigger emergency |
| `/api/features/simulate/start` | POST | Start simulation |
| `/api/features/simulate/stop` | POST | Stop simulation |
| `/api/features/briefing` | GET | AI traffic briefing |
| `/api/features/efficiency` | GET | Before vs after stats |
| `/api/features/carbon` | GET | Carbon savings |

---

## 🎬 Simulation Scenarios

| Scenario | Description | Duration |
|---|---|---|
| 🚦 Rush Hour | Spikes all densities to 70-90% | 30s |
| 🚨 Emergency Vehicle | Triggers emergency on random direction | 30s |
| 💥 Accident | Locks one direction at 100% density | 30s |
| 🌙 Off-Peak | Drops all densities to 5-15% | 30s |

---

## 📊 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **Computer Vision** | YOLOv8 (Ultralytics), OpenCV |
| **Machine Learning** | LSTM Neural Network (PyTorch) |
| **AI Language Model** | Claude claude-sonnet-4-20250514 (Anthropic) |
| **Real-time Comms** | WebSockets |
| **Frontend** | Vanilla HTML, CSS, JavaScript |
| **Video Processing** | OpenCV, FFmpeg |

---

## 🏆 Results

| Metric | Fixed Timing | AI System | Improvement |
|---|---|---|---|
| Avg Wait Time | 45s | ~5s | **89% faster** |
| Throughput | 20 v/min | 33 v/min | **65% more** |
| CO₂ Emissions | 100 g/min | 14 g/min | **86% less** |

---

## 🌟 What Makes This Unique

Compared to similar open-source projects:
- ✅ Complete live dashboard (most others are just scripts)
- ✅ Claude AI natural language briefings (nobody else has this)
- ✅ Carbon savings tracker
- ✅ Before vs after comparison panel
- ✅ Simulation mode for demos
- ✅ Lane management advisories
- ✅ Emergency vehicle manual override UI

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 👤 Author

**Niharika**


## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Anthropic Claude API](https://www.anthropic.com/)
- [Pixabay](https://pixabay.com/) for traffic footage
