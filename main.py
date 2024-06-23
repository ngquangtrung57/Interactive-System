import os
import streamlit as st
import threading
import queue
import cv2
import face_recognition
import numpy as np
import base64
from deepface import DeepFace
import sqlite3
from langchain_helper import LangChainHelper
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

attributes_queue = queue.Queue()
frame_queue = queue.Queue()

known_face_encodings = []
known_face_names = []
known_faces_dir = "user"

# Load known faces from the folder
def load_known_faces():
    logging.info("Loading known faces")
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(known_faces_dir, filename)
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:  
                encoding = encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])
    logging.info("Known faces loaded")

load_known_faces()

face_locations = []
face_names = []
frame = None
unknown_user_counter = len([f for f in os.listdir(known_faces_dir) if f.startswith("unknown_user_")])
attributes_cache = {} 
last_detected_faces = {} 
emotion_detection_interval = 100
age_gender_detection_interval = 100

def capture_frames():
    global frame, video_capture
    while True:
        logging.info("Attempting to open video capture device")
        video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)   
        if not video_capture.isOpened():
            logging.error("Failed to open video capture device") 
            continue
        logging.info("Video capture device opened successfully")
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                logging.error("Failed to capture frame")
                video_capture.release()
                break
            frame_queue.put(frame)

def face_detection_thread():
    global frame, face_locations, face_names, frame_count, unknown_user_counter
    frame_count = 0
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None or frame.size == 0:
                continue
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            frame_count += 1
            if frame_count % 10 == 0: # could modify the frequency here
                logging.info("Processing frame for face recognition")
                new_face_locations = face_recognition.face_locations(rgb_small_frame)
                if new_face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, new_face_locations)
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
                            logging.info("Saving unknown user frame")
                            unknown_user_counter += 1
                            unknown_filename = f"unknown_user_{unknown_user_counter}.jpg"
                            cv2.imwrite(os.path.join(known_faces_dir, unknown_filename), frame)
                            unknown_img = face_recognition.load_image_file(os.path.join(known_faces_dir, unknown_filename))
                            unknown_encoding = face_recognition.face_encodings(unknown_img)[0]
                            known_face_encodings.append(unknown_encoding)
                            known_face_names.append(unknown_filename.split('.')[0])
                            name = unknown_filename.split('.')[0]
                        new_face_names.append(name)
                        last_detected_faces[name] = face_encoding
                        if frame_count % age_gender_detection_interval == 0 or name not in attributes_cache:
                            logging.info(f"Analyzing attributes for {name}")
                            top, right, bottom, left = face_location
                            face_frame = rgb_small_frame[top:bottom, left:right]
                            if face_frame.size == 0:
                                continue
                            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)
                            _, buffer = cv2.imencode('.jpg', face_frame)
                            face_base64 = base64.b64encode(buffer).decode('utf-8')
                            face_data = f"data:image/jpeg;base64,{face_base64}"
                            analysis = DeepFace.analyze(img_path=face_data, actions=['age', 'gender', 'emotion'], enforce_detection=False, prog_bar=False)
                            attributes_cache[name] = {
                                'gender': analysis['gender'],
                                'age': analysis['age'],
                                'emotion': analysis['dominant_emotion']
                            }
                        else:
                            logging.info(f"Updating emotion for {name}")
                            top, right, bottom, left = face_location
                            face_frame = rgb_small_frame[top:bottom, left:right]
                            if face_frame.size == 0:
                                continue
                            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)
                            _, buffer = cv2.imencode('.jpg', face_frame)
                            face_base64 = base64.b64encode(buffer).decode('utf-8')
                            face_data = f"data:image/jpeg;base64,{face_base64}"
                            analysis = DeepFace.analyze(img_path=face_data, actions=['emotion'], enforce_detection=False, prog_bar=False)
                            attributes_cache[name]['emotion'] = analysis['dominant_emotion']
                        attributes_info.append(
                            f"Name: {name}, Gender: {attributes_cache[name]['gender']}, Age: {attributes_cache[name]['age']}, Emotion: {attributes_cache[name]['emotion']}"
                        )
                    face_locations = new_face_locations
                    face_names = new_face_names
                    attributes_queue.put(attributes_info)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            frame_queue.put(frame)

def display_webcam():
    logging.info("Starting webcam display")
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is not None and frame.size != 0:
                cv2.imshow("Webcam Feed", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()

def conversation_thread():
    logging.info("Starting conversation thread")

    user_id = st.text_input("Enter your user ID", "test_user")

    conversation_history = st.empty()
    user_info_text = st.empty()
    response_status_text = st.empty()

    if "conversation_started" not in st.session_state:
        st.session_state["conversation_started"] = False
        st.session_state["pause"] = False
        st.session_state["generating_response"] = False  
        st.session_state["conversation_history"] = []

    if not st.session_state["conversation_started"] and st.button("Start"):
        logging.info("Starting conversation")
        st.session_state["conversation_started"] = True
        st.session_state["pause"] = False
        st.session_state["generating_response"] = False 
        st.session_state["conversation_history"] = []
        helper.clear_memory()
        
        greeting = "Hello, how can I help you today?"
        st.session_state["conversation_history"].append(f"🤖 AI: {greeting}")
        conversation_history.markdown("\n".join(st.session_state["conversation_history"]))
        helper.play_audio_chunks(greeting)

        # End button
        if st.button("End"):
            logging.info("Ending conversation")
            st.session_state["conversation_started"] = False
            st.session_state["pause"] = False
            st.write("Conversation ended.")
            helper.clear_memory()
            conn.commit()
            st.experimental_rerun() 

    # Capture and process user input
    while st.session_state["conversation_started"] and not st.session_state["pause"]:
        user_input = helper.capture_user_input()
        if user_input and not st.session_state["generating_response"]:
            logging.info(f"Captured user input: {user_input}")
            st.session_state["conversation_history"].append(f"{user_id}: {user_input}")

            # Update conversation display with user input
            conversation_history.markdown("\n".join(st.session_state["conversation_history"]))

            # Indicate the bot is generating a response
            response_status_text.text("🤖 Getting response from LLM...")
            st.session_state["generating_response"] = True 

            # Fetch attributes from the queue
            if not attributes_queue.empty():
                attributes_info = attributes_queue.get()
                user_info_text.text("\n".join(attributes_info))
                name, gender, age, emotion = attributes_info[0].split(',')[0].split(': ')[1], attributes_info[0].split(',')[1].split(': ')[1], int(attributes_info[0].split(',')[2].split(': ')[1]), attributes_info[0].split(',')[3].split(': ')[1]
            else:
                # Default mock data if no attributes detected
                gender = "male"
                age = 25
                emotion = "happy"

            # Generate response
            response = helper.get_response(user_id=user_id, gender=gender, age=age, emotion=emotion, cursor=cursor, user_input=user_input)
            logging.info(f"Generated response: {response}")
            st.session_state["conversation_history"].append(f"🤖 AI: {response}")

            # Update conversation display with AI response
            conversation_history.markdown("\n".join(st.session_state["conversation_history"]))
            response_status_text.text("")  
            st.session_state["generating_response"] = False  
        else:
            st.write("Could not capture your input. Please try again.")

st.title("Interactive Face Recognition and Conversation System")
attributes_text = st.empty()

def update_ui():
    logging.info("Starting UI update thread")
    while True:
        if not attributes_queue.empty():
            attributes_info = attributes_queue.get()
            attributes_text.text("\n".join(attributes_info))
            st.rerun()

if __name__ == "__main__":
    helper = LangChainHelper()
    conn = sqlite3.connect('chat_memory.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_memory (user_id TEXT PRIMARY KEY, conversation TEXT)''')
    conn.commit()

    logging.info("Starting threads")
    threading.Thread(target=capture_frames, daemon=True).start()
    threading.Thread(target=face_detection_thread, daemon=True).start()
    threading.Thread(target=display_webcam, daemon=True).start()

    conversation_thread()
