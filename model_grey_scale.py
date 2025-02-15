import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLO model
model = YOLO(r"C:\Users\shrav\Downloads\archive\runs\detect\train\weights\best.pt")

# Run detection
image_path = r"C:\Users\shrav\Downloads\oil_tanker_cushing.png"
results = model(image_path, conf=0.4)  # Run detection

# Load original image
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Apply histogram equalization to enhance contrast
image_gray = cv2.equalizeHist(image_gray)

# Create a blank mask for shadows
shadow_mask = np.zeros_like(image_gray)

# Process each detected tank separately
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box coordinates
        tank_roi = image_gray[y1:y2, x1:x2]  # Extract the detected tank region

        # Apply Gaussian blur to remove sharp edges (helps remove outer shadow)
        blurred_tank = cv2.GaussianBlur(tank_roi, (5, 5), 0)

        # Use adaptive thresholding to detect **only the dark regions inside the tank**
        tank_shadow = cv2.adaptiveThreshold(blurred_tank, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 25, 5)

        # Place detected inner shadow back in the correct location on the full image
        shadow_mask[y1:y2, x1:x2] = tank_shadow

# Show grayscale image with detected inner shadows
plt.figure(figsize=(10, 6))
plt.imshow(shadow_mask, cmap="gray")
plt.title("Detected Oil Tank Shadows (Inner Region Only)")
plt.axis("off")
plt.show()