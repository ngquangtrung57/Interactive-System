import cv2
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

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

        # Perform face analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)

        # Extract attributes
        age = str(result[0]['age'])
        gender = str(result[0]['gender'])
        race = str(result[0]['dominant_race'])
        emotion = result[0]['dominant_emotion']

        # Draw rectangle around face and label with predicted attributes
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"Age: {age}\nGender: {gender}\nRace: {race}\nEmotion: {emotion}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
