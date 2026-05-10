import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('videos/traffic.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Total frames:', total)

COCO_CLASSES = {
    0:'person', 1:'bicycle', 2:'car', 3:'motorcycle',
    4:'airplane', 5:'bus', 6:'train', 7:'truck', 8:'boat',
    14:'bird', 15:'cat', 16:'dog', 58:'potted plant'
}

cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, frame = cap.read()
frame = cv2.resize(frame, (1280, 720))
cv2.imwrite('test_frame.jpg', frame)
print('Saved test_frame.jpg — open this file to see what the video looks like')

r = model(frame, conf=0.1, verbose=False)[0]
print(f'Total detections at conf=0.1: {len(r.boxes) if r.boxes else 0}')
if r.boxes:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = COCO_CLASSES.get(cls_id, f'class_{cls_id}')
        print(f'  {name} (class {cls_id}) conf={conf:.2f}')

cap.release()