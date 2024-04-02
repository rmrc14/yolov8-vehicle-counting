from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.track(source="https://youtu.be/Zgi9g1ksQHc", conf=0.3, iou=0.5, show=True) 
