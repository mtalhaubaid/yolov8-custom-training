from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load your YOLO model
model = YOLO("yolov8n.pt")

# Initialize the class mapping dictionary
class_mapping_list = {}

# Load class mappings from a text file
class_mapping_file = "class_mappings.txt"
if os.path.exists(class_mapping_file):
    with open(class_mapping_file, "r") as file:
        for line in file:
            class_id, file_name = line.strip().split(": ")
            class_mapping_list[int(class_id)] = file_name

# Get the last assigned class ID
last_class_id = max(class_mapping_list.keys()) if class_mapping_list else -1

# Ask the user for the file name on the first frame
file_name = input("Enter the file name (for both .jpg and .txt files): ")

# Initialize a set to track detected class names
detected_classes = set()

# Open the video file
video_path = "mouse.mp4"
cap = cv2.VideoCapture(video_path)
image_id = 1

# Flag to indicate whether the first class has been detected
first_detection = True

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    detections = model.predict(frame, conf=0.65, show_labels=False, boxes=False)

    for detection in detections:
        class_name = detection.names[0]  # Get the detected class name

        if first_detection:
            # Create a folder for the detected class from the first detection
            annotation_folder = f"anno/{class_name}"
            os.makedirs(annotation_folder, exist_ok=True)
            first_detection = False

        if class_name in class_mapping_list.values():
            # Find the class ID for the entered class name
            class_id = next(key for key, value in class_mapping_list.items() if value == class_name)
        else:
            if file_name not in class_mapping_list.values():
                # Assign the next available class ID if it's a new class
                last_class_id += 1
                class_id = last_class_id
                # Add the new class to the dictionary
                class_mapping_list[class_id] = file_name

        # Define the annotation path and image path using the user-provided file name and image_id
        annotation_path = f"{annotation_folder}/{file_name}_{image_id}.txt"
        annotated_image_path = f"{annotation_folder}/{file_name}_{image_id}.jpg"

        with open(annotation_path, 'w') as f:
            box = detection.boxes
            xyxy = box.xyxy.tolist()
            class_ids = box.cls.tolist()

            for i in range(len(class_ids)):
                confidence = box.conf[i]
                x1, y1, x2, y2 = map(int, xyxy[i])

                x_center = (xyxy[i][0] + xyxy[i][2]) / 2 / frame.shape[1]
                y_center = (xyxy[i][1] + xyxy[i][3]) / 2 / frame.shape[0]
                width = (xyxy[i][2] - xyxy[i][0]) / frame.shape[1]
                height = (xyxy[i][3] - xyxy[i][1]) / frame.shape[0]

                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                f.write(line)

                # Draw bounding box on the image
                class_label = f"{class_name} {confidence}"
                color = (0, 255, 0)  # Green color for the bounding box
                thickness = 2  # Thickness of the bounding box lines
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness)

    # Save the annotated image and increment image_id
    cv2.imwrite(annotated_image_path, frame)
    image_id += 1

# Save the updated class mappings to the text file
with open(class_mapping_file, "w") as file:
    for class_id, class_name in class_mapping_list.items():
        file.write(f"{class_id}: {class_name}\n")

