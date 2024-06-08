import cv2
from deepface import DeepFace
import numpy as np

user_credentials = {
    'user_id': ['test'],
    'password': ['password']
}

user_id = input("Please enter your user_id: ")
password = input("Please enter your password: ")

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

face_identities = {}

# Check if the entered credentials are correct
if user_id in user_credentials['user_id'] and user_credentials['password'][user_credentials['user_id'].index(user_id)] == password:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform face analysis only if logged in successfully
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Extract attributes
            emotion = result[0]['dominant_emotion']

            # Store face identity
            face_identities[(x, y, w, h)] = {'emotion': emotion}

            # Draw rectangle around face and label with predicted attributes
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, face_identities[(x, y, w, h)]['emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)

        # Press 'q' to exit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
else:
    print("Please create an account before preceed")
    new_user_id = input("Please enter your user id:")
    new_password = input("Please enter your password:")
    user_credentials['password'].append(new_password)
    user_credentials['user_id'].append(new_user_id)

    condition = True 

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if condition:
            for (x, y, w, h) in faces:
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Perform face analysis only if logged in successfully
                result = DeepFace.analyze(face_roi, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)

                # Extract attributes
                emotion = result[0]['dominant_emotion']

                # Store face identity
                face_identities[(x, y, w, h)] = {'emotion': emotion}

                # Draw rectangle around face and label with predicted attributes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, face_identities[(x, y, w, h)]['emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                condition = False

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]

            # Perform face analysis only if logged in successfully
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Extract attributes
            emotion = result[0]['dominant_emotion']

            # Store face identity
            face_identities[(x, y, w, h)] = {'emotion': emotion}

            # Draw rectangle around face and label with predicted attributes
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, face_identities[(x, y, w, h)]['emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)

        # Press 'q' to exit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    

# Release the capture
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()