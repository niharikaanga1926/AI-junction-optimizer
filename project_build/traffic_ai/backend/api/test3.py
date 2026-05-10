import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('videos/traffic.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Total frames:', total)

VEHICLE_CLASSES = {2, 3, 5, 7}

for pct in [10, 25, 50, 75, 90]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pct / 100))
    ret, frame = cap.read()
    if not ret:
        print(f'  {pct}% — could not read frame')
        continue
    frame = cv2.resize(frame, (1280, 720))
    r = model(frame, conf=0.25, verbose=False)[0]
    vehicles = sum(1 for b in r.boxes if int(b.cls[0]) in VEHICLE_CLASSES) if r.boxes else 0
    all_det = len(r.boxes) if r.boxes else 0
    print(f'  {pct}% through video: {vehicles} vehicles, {all_det} total detections')

cap.release()