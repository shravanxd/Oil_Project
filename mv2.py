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

# Create a blank mask for shadow extraction
shadow_mask = np.zeros_like(image_gray)

# Sunlight direction assumption (modify based on actual conditions)
sunlight_angle = 15  # Assume sunlight comes from top-right (45 degrees)

# Process each detected tank
for result in results:
    for box in result.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box coordinates

        # Calculate tank diameter & center
        tank_diameter = min(x2 - x1, y2 - y1)  # Use the smaller side as the diameter
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Tank center

        # Define the shadow search region (offset in sunlight direction)
        shadow_x1 = int(cx + 0.2 * tank_diameter * np.cos(np.radians(sunlight_angle)))
        shadow_y1 = int(cy + 0.2 * tank_diameter * np.sin(np.radians(sunlight_angle)))
        shadow_x2 = int(cx + 0.5 * tank_diameter * np.cos(np.radians(sunlight_angle)))
        shadow_y2 = int(cy + 0.5 * tank_diameter * np.sin(np.radians(sunlight_angle)))

        # Extract the potential shadow region
        shadow_region = image_gray[max(0, shadow_y1):min(image_gray.shape[0], shadow_y2),
                                   max(0, shadow_x1):min(image_gray.shape[1], shadow_x2)]

        if shadow_region.size == 0:
            continue  # Skip if the region is empty

        # Apply Gaussian blur to remove sharp edges
        shadow_region_blurred = cv2.GaussianBlur(shadow_region, (5, 5), 0)

        # Detect darkest region using thresholding
        _, tank_shadow = cv2.threshold(shadow_region_blurred, np.min(shadow_region_blurred) + 20,
                                       255, cv2.THRESH_BINARY_INV)

        # Place the detected shadow on the full mask
        shadow_mask[max(0, shadow_y1):min(image_gray.shape[0], shadow_y2),
                    max(0, shadow_x1):min(image_gray.shape[1], shadow_x2)] = tank_shadow

# Show final detected inner shadow regions
plt.figure(figsize=(10, 6))
plt.imshow(shadow_mask, cmap="gray")
plt.title("Optimized Inner Shadow Detection (Sunlight-Aware)")
plt.axis("off")
plt.show()