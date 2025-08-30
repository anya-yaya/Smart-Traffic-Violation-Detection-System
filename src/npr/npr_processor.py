# src/npr/npr_processor.py

import easyocr
import cv2
import numpy as np
import os
from ultralytics import YOLO

class NumberPlateRecognizer:
    def __init__(self, languages=['en'], license_plate_class_id=2): # Assuming class ID 2 for license plate (often custom)
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(languages, gpu=False) # Set gpu=True if you have a CUDA-enabled GPU
        self.license_plate_class_id = license_plate_class_id
        self.yolo_model = YOLO('yolov8n.pt') # Using YOLO to detect number plates

    def extract_plate(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return "N/A"

        # Use YOLO to detect potential license plate regions first
        results = self.yolo_model(image, verbose=False)
        detections = results[0].boxes.data.tolist()

        plate_text = "N/A"
        best_conf = 0

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            # Check if the detected object is a license plate (you might need a custom trained model)
            # For a general YOLO model like yolov8n, you'll need to rely on filtering objects.
            # A common heuristic is to check for small, rectangular objects with aspect ratio typical of plates.
            # Or, ideally, you'd fine-tune YOLO for license plates.

            # Heuristic for typical license plate aspect ratio (adjust as needed)
            aspect_ratio = (x2 - x1) / (y2 - y1)
            if 2 < aspect_ratio < 6 and conf > 0.5: # and int(cls) == self.license_plate_class_id (if you have a custom class)
                # Crop the region of interest (ROI)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                plate_roi = image[y1:y2, x1:x2]

                if plate_roi.shape[0] == 0 or plate_roi.shape[1] == 0:
                    continue

                # Perform OCR on the ROI
                ocr_results = self.reader.readtext(plate_roi)

                for (bbox, text, score) in ocr_results:
                    # Simple filtering for plausible plate text (alphanumeric, length)
                    cleaned_text = "".join(filter(str.isalnum, text)).upper()
                    if len(cleaned_text) >= 4 and score > best_conf: # Min 4 chars for a plate, and higher confidence
                        plate_text = cleaned_text
                        best_conf = score
                        break # Take the best one found

        return plate_text

# Example usage (for testing purposes)
#if __name__ == "__main__":
  #  recognizer = NumberPlateRecognizer()
  #  plate_text = recognizer.extract_plate("car.jpg")  # Replace with your test image
   # print("Detected Number Plate:", plate_text)


