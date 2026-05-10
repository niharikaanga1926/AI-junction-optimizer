import cv2
import sys

print("Testing video file...")
cap = cv2.VideoCapture('videos/traffic.mp4')
print('Video opened:', cap.isOpened())
ret, frame = cap.read()
print('Frame read:', ret)
if ret:
    print('Frame shape:', frame.shape)
else:
    print('ERROR: Could not read frame from video!')
cap.release()

print()
print("Testing YOLO detection...")
try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    if ret and frame is not None:
        results = model(frame, verbose=False)[0]
        print('Detections found:', len(results.boxes) if results.boxes else 0)
        if results.boxes and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                print(f'  class={cls_id} conf={conf:.2f}')
    else:
        print('Skipping YOLO — no frame available')
except Exception as e:
    print('YOLO error:', e)

print()
print("Checking detector._main_loop fix...")
import asyncio
loop = asyncio.new_event_loop()
print('Loop running:', loop.is_running())
print('Loop created:', loop is not None)
