# src/detection/yolo_detector.py

from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import os

class ViolationDetector:
    def __init__(self, yolo_model_path='models/yolov8n.pt'):
        # Ensure the directory for the model exists or handle download
        if not os.path.exists(yolo_model_path):
            print(f"YOLO model not found at {yolo_model_path}. Attempting to download yolov8n.")
            self.model = YOLO('yolov8n.pt') # This will download the nano model if not present
        else:
            self.model = YOLO(yolo_model_path)

        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Define classes for detection (adjust based on your YOLO model's capabilities)
        self.person_class_id = 0 # Default for COCO dataset
        self.motorcycle_class_id = 3 # Default for COCO dataset
        self.bicycle_class_id = 1 # Default for COCO dataset
        self.cell_phone_class_id = 67 # Default for COCO dataset
        self.helmet_class_id = None # No default helmet class in standard YOLOv8. Will rely on pose for helmet detection.

        # Thresholds for violation detection (tune these)
        self.HEAD_HELMET_OVERLAP_THRESHOLD = 0.2 # Percentage of head covered by helmet for detection
        self.PHONE_DISTANCE_THRESHOLD = 0.15 # Normalized distance for phone to be near ear/face
        self.MIN_TRIPLE_RIDING_PEOPLE = 3 # Minimum people on a single motorcycle for triple riding


    def detect(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return [], image

        violations_detected = []
        annotated_image = image.copy()
        h, w, _ = image.shape

        # 1. YOLO Object Detection
        results = self.model(image, verbose=False)
        detections = results[0].boxes.data.tolist() # xyxy, conf, cls

        # Group detections by vehicle
        vehicle_detections = [] # Stores (bbox, conf, cls, riders)
        person_detections = []  # Stores (bbox, conf, cls)
        phone_detections = [] # Stores (bbox, conf, cls)
        helmet_detections = [] # Stores (bbox, conf, cls) if you have a custom trained model for helmets

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == self.motorcycle_class_id or int(cls) == self.bicycle_class_id:
                vehicle_detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf, 'cls': cls, 'riders': []})
            elif int(cls) == self.person_class_id:
                person_detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf, 'cls': cls})
            elif int(cls) == self.cell_phone_class_id:
                phone_detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf, 'cls': cls})
            # Add logic for helmet detection if self.helmet_class_id is set

        # Associate persons with vehicles (simple proximity check)
        for person_det in person_detections:
            px1, py1, px2, py2 = person_det['bbox']
            person_center_x = (px1 + px2) / 2
            person_center_y = (py1 + py2) / 2

            closest_vehicle = None
            min_dist = float('inf')

            for vehicle_det in vehicle_detections:
                vx1, vy1, vx2, vy2 = vehicle_det['bbox']
                vehicle_center_x = (vx1 + vx2) / 2
                vehicle_center_y = (vy1 + vy2) / 2

                dist = np.sqrt((person_center_x - vehicle_center_x)**2 + (person_center_y - vehicle_center_y)**2)
                # Check if person is within vehicle's bounding box or very close below/above it
                if px1 >= vx1 and px2 <= vx2 and py2 > vy1: # Person is within vehicle width and below vehicle top
                     if dist < min_dist:
                        min_dist = dist
                        closest_vehicle = vehicle_det

            if closest_vehicle:
                closest_vehicle['riders'].append(person_det)


        # 2. MediaPipe Pose and Hands Estimation for detailed analysis
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)

        # Process each vehicle and its associated riders
        for vehicle_det in vehicle_detections:
            vx1, vy1, vx2, vy2 = [int(c) for c in vehicle_det['bbox']]
            num_riders = len(vehicle_det['riders'])

            # Triple Riding Detection
            if num_riders >= self.MIN_TRIPLE_RIDING_PEOPLE:
                violations_detected.append("Triple Riding")
                cv2.rectangle(annotated_image, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2) # Red for violation
                cv2.putText(annotated_image, "Triple Riding", (vx1, vy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


            for rider_det in vehicle_det['riders']:
                px1, py1, px2, py2 = [int(c) for c in rider_det['bbox']]
                rider_image = image[py1:py2, px1:px2]
                if rider_image.shape[0] == 0 or rider_image.shape[1] == 0:
                    continue # Skip empty images

                rider_image_rgb = cv2.cvtColor(rider_image, cv2.COLOR_BGR2RGB)
                rider_pose_results = self.pose.process(rider_image_rgb)
                rider_hands_results = self.hands.process(rider_image_rgb)

                # Not Wearing Helmet Detection (using pose estimation)
                if rider_pose_results.pose_landmarks:
                    nose = rider_pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                    left_ear = rider_pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
                    right_ear = rider_pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]

                    # Convert normalized coordinates to image coordinates
                    nose_img = (int(nose.x * rider_image.shape[1]), int(nose.y * rider_image.shape[0]))
                    left_ear_img = (int(left_ear.x * rider_image.shape[1]), int(left_ear.y * rider_image.shape[0]))
                    right_ear_img = (int(right_ear.x * rider_image.shape[1]), int(right_ear.y * rider_image.shape[0]))

                    head_center_x = (left_ear_img[0] + right_ear_img[0]) // 2
                    head_center_y = (nose_img[1] + max(left_ear_img[1], right_ear_img[1])) // 2 # Approx head top

                    # A very simple heuristic: If the head is detected, assume no helmet if no specific helmet object is detected
                    # A more robust system would involve a separate helmet detection model or more complex landmark analysis.
                    # For now, we assume if a person's head is clearly visible and not obscured, they might not be wearing a helmet.
                    # If you have a helmet class ID, you'd check for overlap between head landmarks and helmet bbox.
                    helmet_present = False
                    if self.helmet_class_id:
                         # Check if any helmet detection overlaps with the person's head region
                         for h_det in helmet_detections:
                             hx1, hy1, hx2, hy2 = h_det['bbox']
                             # Simple overlap check
                             if max(hx1, px1) < min(hx2, px2) and max(hy1, py1) < min(hy2, py2):
                                 # Calculate intersection over union (IoU) or just area overlap
                                 # and check if significant portion of head is covered.
                                 intersection_area = max(0, min(hx2, px2) - max(hx1, px1)) * max(0, min(hy2, py2) - max(hy1, py1))
                                 person_area = (px2 - px1) * (py2 - py1)
                                 if person_area > 0 and intersection_area / person_area > self.HEAD_HELMET_OVERLAP_THRESHOLD:
                                     helmet_present = True
                                     break

                    if not helmet_present and pose_results.pose_landmarks:
                         violations_detected.append("Not Wearing Helmet")
                         cv2.rectangle(annotated_image, (px1, py1), (px2, py2), (0, 165, 255), 2) # Orange for violation
                         cv2.putText(annotated_image, "No Helmet", (px1, py1 + 15),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                # Using Phone While Riding Detection (using hands estimation)
                if rider_hands_results.multi_hand_landmarks:
                    for hand_landmarks in rider_hands_results.multi_hand_landmarks:
                        # Draw hand landmarks (optional)
                        # self.mp_drawing.draw_landmarks(annotated_image[py1:py2, px1:px2], hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # Check if any phone object is close to hand landmarks
                        for phone_det in phone_detections:
                            ph_x1, ph_y1, ph_x2, ph_y2 = phone_det['bbox']
                            phone_center_x = (ph_x1 + ph_x2) / 2
                            phone_center_y = (ph_y1 + ph_y2) / 2

                            # Check distance of phone center to any hand landmark (simple approach)
                            for landmark in hand_landmarks.landmark:
                                lx, ly = int(landmark.x * rider_image.shape[1]) + px1, int(landmark.y * rider_image.shape[0]) + py1
                                distance = np.sqrt((phone_center_x - lx)**2 + (phone_center_y - ly)**2)
                                # Normalize distance by image diagonal for robustness
                                normalized_distance = distance / np.sqrt(w**2 + h**2)

                                if normalized_distance < self.PHONE_DISTANCE_THRESHOLD:
                                    violations_detected.append("Using Phone While Riding")
                                    cv2.rectangle(annotated_image, (px1, py1), (px2, py2), (0, 255, 255), 2) # Yellow for violation
                                    cv2.putText(annotated_image, "Using Phone", (px1, py1 + 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                    break # Only detect once per person
                            if "Using Phone While Riding" in violations_detected:
                                break
                        if "Using Phone While Riding" in violations_detected:
                                break
                # Draw bounding boxes for persons
                cv2.rectangle(annotated_image, (px1, py1), (px2, py2), (255, 0, 0), 2) # Blue for detected person
                cv2.putText(annotated_image, "Rider", (px1, py1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw bounding boxes for vehicles
        for vehicle_det in vehicle_detections:
            vx1, vy1, vx2, vy2 = [int(c) for c in vehicle_det['bbox']]
            cv2.rectangle(annotated_image, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2) # Green for detected vehicle
            cv2.putText(annotated_image, f"Vehicle ({len(vehicle_det['riders'])} riders)", (vx1, vy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        return list(set(violations_detected)), annotated_image # Return unique violations