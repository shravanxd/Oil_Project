
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO(r"C:\Users\shrav\Downloads\archive\runs\detect\train\weights\best.pt")

# Run inference
image_path = r"C:\Users\shrav\Downloads\test1.png"
results = model(image_path, conf=0.4)  # Run detection

# Load original image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Draw bounding boxes (without labels)
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box coordinates
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

# Show image
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.axis("off")
plt.show()
