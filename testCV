import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import time
import threading
from deepface import DeepFace
import base64
from PIL import Image

known_face_encodings = []
known_face_names = []
known_faces_dir = "user"

# Function to load known faces from the folder
def load_known_faces():
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(known_faces_dir, filename)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:  # Ensure that an encoding was found
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])

load_known_faces()

face_locations = []
face_names = []
frame = None
unknown_user_counter = len([f for f in os.listdir(known_faces_dir) if f.startswith("unknown_user_")])
attributes_cache = {}  # Cache for storing detected attributes
last_detected_faces = {}  # Cache to store the last detected face encodings and names
emotion_detection_interval = 100
age_gender_detection_interval = 100

st.title("Face Recognition App")
stframe = st.empty()
matched_names_text = st.empty()
attributes_text = st.empty()

video_capture = cv2.VideoCapture(0)

def capture_frames():
    global frame
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

frame_count = 0

while True:
    if frame is not None:
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        frame_count += 1
        if frame_count % 30 == 0:  # Process every 10th frame for face recognition
            new_face_locations = face_recognition.face_locations(rgb_small_frame)
            print(f"Detected face locations: {new_face_locations}")  # Debug print

            if new_face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, new_face_locations)
                print(f"Computed face encodings: {face_encodings}")  # Debug print

                new_face_names = []
                attributes_info = []

                for face_encoding, face_location in zip(face_encodings, new_face_locations):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    name = "Unknown"

                    if matches:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                    else:
                        # Save unknown user frame
                        unknown_user_counter += 1
                        unknown_filename = f"unknown_user_{unknown_user_counter}.jpg"
                        cv2.imwrite(os.path.join(known_faces_dir, unknown_filename), frame)
                        
                        # Load the new unknown face
                        unknown_img = face_recognition.load_image_file(os.path.join(known_faces_dir, unknown_filename))
                        unknown_encoding = face_recognition.face_encodings(unknown_img)[0]
                        known_face_encodings.append(unknown_encoding)
                        known_face_names.append(unknown_filename.split('.')[0])
                        
                        name = unknown_filename.split('.')[0]

                    new_face_names.append(name)
                    last_detected_faces[name] = face_encoding

                    # Analyze the detected face every 50 frames or if not previously detected
                    if frame_count % age_gender_detection_interval == 0 or name not in attributes_cache:
                        top, right, bottom, left = face_location
                        face_frame = rgb_small_frame[top:bottom, left:right]
                        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for DeepFace

                        _, buffer = cv2.imencode('.jpg', face_frame)
                        face_base64 = base64.b64encode(buffer).decode('utf-8')
                        face_data = f"data:image/jpeg;base64,{face_base64}"

                        analysis = DeepFace.analyze(img_path=face_data, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                        attributes_cache[name] = {
                            'gender': analysis['gender'],
                            'age': analysis['age'],
                            'emotion': analysis['dominant_emotion']
                        }
                    else:
                        # Only update emotion if age and gender are already detected
                        top, right, bottom, left = face_location
                        face_frame = rgb_small_frame[top:bottom, left:right]
                        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for DeepFace

                        _, buffer = cv2.imencode('.jpg', face_frame)
                        face_base64 = base64.b64encode(buffer).decode('utf-8')
                        face_data = f"data:image/jpeg;base64,{face_base64}"

                        analysis = DeepFace.analyze(img_path=face_data, actions=['emotion'], enforce_detection=False)
                        attributes_cache[name]['emotion'] = analysis['dominant_emotion']

                    attributes_info.append(
                        f"Name: {name}, Gender: {attributes_cache[name]['gender']}, Age: {attributes_cache[name]['age']}, Emotion: {attributes_cache[name]['emotion']}"
                    )

                # Update the face locations and names only if detection is successful
                face_locations = new_face_locations
                face_names = new_face_names

                # Display the matched names and attributes
                matched_names_text.text("Matched names: " + ", ".join(face_names))
                attributes_text.text("\n".join(attributes_info))

        # Display the results for every frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        stframe.image(frame, channels="BGR")

video_capture.release()
capture_thread.join()
