import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO

# Load YOLOv8 model (ensure you have this model file)
model = YOLO(r"D:\yolov8\yolov8n-face.pt")  # Replace with your model file path# Load known face encodings and names
known_face_encodings = np.load(r"D:\yolov8\face_recognition\encodings.npy", allow_pickle=True)
known_face_names = np.load(r"D:\yolov8\face_recognition\names.npy", allow_pickle=True)

# Function to detect faces using YOLOv8
def detect_faces(image):
    results = model(image)
    faces = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            faces.append((x1, y1, x2, y2))
    return faces

# Function to recognize faces
def recognize_faces(image, face_locations):
    face_encodings = face_recognition.face_encodings(image, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    return face_names

def main():
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture a single frame of video
        ret, frame = video_capture.read()

        # Detect faces
        face_locations = detect_faces(frame)
        
        # Convert face locations to the format required by face_recognition
        face_recognition_locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in face_locations]
        
        # Recognize faces
        face_names = recognize_faces(frame, face_recognition_locations)
        
        # Display the results
        for (x1, y1, x2, y2), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(name), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Ensure 'name' is a string

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()