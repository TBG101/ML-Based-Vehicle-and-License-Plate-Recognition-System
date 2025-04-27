import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import sys
import numpy as np
from PIL import Image

PLATE_MODEL_PATH = "models/plate_recognition_yolo_model.pt"


def get_license_plates(debug=False, image=None):
    model = YOLO(PLATE_MODEL_PATH)
    # Load and preprocess image
    if debug:
        image = cv2.imread("license-plates.jpg")

    if image is None:
        print("Error: Image file not found or unable to load.")
        sys.exit(1)

    # Convert PIL image to OpenCV format
    if not isinstance(image, np.ndarray):
        image = Image.open(image)  # Open as PIL image
        image = np.array(image)  # Convert to NumPy array
        # Convert to OpenCV format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = model(image)

    # Check if any objects are detected
    if not results or len(results[0].boxes) == 0:
        print("No objects detected.")

    plate_images = []
    # Loop through detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()

            if confidence > 0.5:  # Lower confidence threshold
                # Draw a rectangle (square) around the detected plate
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Extract the plate image
                plate_image = image[y1:y2, x1:x2]
                plate_images.append(plate_image)

    return image,  plate_images
