import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import sys
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

PLATE_MODEL_PATH = "models/plate_recognition_yolo_model.pt"

def extract_text_from_image(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, debug: bool = False) -> str:
    """Extract text from a given image using OCR."""
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    plate = image[y1:y2, x1:x2]
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    cleaned = cv2.adaptiveThreshold(
        plate, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 12
    )

    res = ocr.ocr(cleaned, cls=True)
    print("Printing data")
    nb_plate: str = ""

    for line in res:
        for word in line:
            print(word[1][0], " (Confidence:", word[1][1], ")")
            nb_plate += word[1][0].replace(" ", "") + " "
    if debug:
        plt.imshow(cleaned, cmap="gray")
        plt.axis("off")
        plt.show()

    # Filter to keep only alphanumeric characters and the '|' character
    nb_plate = ''.join(c for c in nb_plate if c.isalnum() or c == '|')
    return nb_plate


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

    all_plates = ""
    # Loop through detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()

            if confidence > 0.5:  # Lower confidence threshold
                plate_text = extract_text_from_image(
                    image, x1, y1, x2, y2, debug=debug)

                # Calculate font scale and thickness based on image size
                thickness = max(1, int(image.shape[1] / 500))

                # Draw bounding box and text
                cv2.rectangle(image, (x1, y1), (x2, y2),
                              (0, 255, 0), thickness)
                # Scale text size based on bounding box height
                text_scale = max(0.5, min((y2 - y1) / 50, 1.5))
                cv2.putText(
                    image,
                    plate_text.strip(),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    text_scale,
                    (0, 255, 0),
                    thickness,
                )
                all_plates += plate_text.strip() + " | "

    # Display result
    if debug:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
    else:
        image = cv2.imencode(".png", image)[1]
        return image, all_plates.strip()
