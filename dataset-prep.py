import os
import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO

# Load YOLOv8 model (ensure you have this model file)
model = YOLO(r"D:\yolov8\yolov8m-face.pt")  # Replace with your model file path

# Path to your dataset directory
dataset_dir = r"D:\yolov8\face_recognition\input"  # Replace with your dataset directory

# Initialize lists to hold encodings and names
known_face_encodings = []
known_face_names = []

# Iterate through each person in the dataset directory
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if os.path.isdir(person_dir):
        for file_name in os.listdir(person_dir):
            if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
                # Load the image
                image_path = os.path.join(person_dir, file_name)
                image = face_recognition.load_image_file(image_path)

                # Detect faces using face_recognition (you can also use YOLO here if preferred)
                face_locations = face_recognition.face_locations(image, model='cnn')  # or model='hog'

                # Encode faces
                face_encodings = face_recognition.face_encodings(image, face_locations)

                for face_encoding in face_encodings:
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)

# Save the encodings and names to disk for later use (optional)
np.save('encodings.npy', known_face_encodings)
np.save('names.npy', known_face_names)
