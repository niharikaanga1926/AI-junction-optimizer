import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('videos/traffic.mp4')

# Skip to middle of video where traffic is likely
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

ret, frame = cap.read()
cap.release()

# Resize like detector does
frame = cv2.resize(frame, (1280, 720))
cv2.imwrite('test_frame.jpg', frame)
print('Frame saved as test_frame.jpg — open it to see what the video looks like')

results = model(frame, conf=0.25, verbose=False)[0]
print('Detections at conf=0.25:', len(results.boxes) if results.boxes else 0)

results2 = model(frame, conf=0.1, verbose=False)[0]
print('Detections at conf=0.10:', len(results2.boxes) if results2.boxes else 0)

if results2.boxes:
    for box in results2.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f'  class={cls_id} conf={conf:.2f}')