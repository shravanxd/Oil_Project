from ultralytics import YOLO

# Load YOLOv8 model (pre-trained on COCO)
model = YOLO("yolov8n.pt")  # Using 'n' (nano) for fast training

# Train the model on your dataset
model.train(data=r"C:\Users\shrav\Downloads\archive\dataset\data.yaml",
            epochs=30, imgsz=640, batch=8)
