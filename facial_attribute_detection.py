import cv2
from deepface import DeepFace
import sqlite3

# Create a SQLite connection and cursor
conn = sqlite3.connect('people_data.db')
cursor = conn.cursor()

# Create a table to store the data without the Count column
cursor.execute('''CREATE TABLE IF NOT EXISTS PeopleData
            (PersonID INTEGER PRIMARY KEY,
             Age TEXT,
             Gender TEXT,
             Race TEXT,
             Emotion TEXT)''')

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize dictionary to store face identities
face_identities = {}

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

        # Perform face analysis only if face is new
        if (x, y, w, h) not in face_identities:
            # Perform face analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)

            # Extract attributes
            age = str(result[0]['age'])
            gender = str(result[0]['dominant_gender'])
            race = str(result[0]['dominant_race'])
            emotion = result[0]['dominant_emotion']

            # Store face identity
            face_identities[(x, y, w, h)] = {'age': age, 'gender': gender, 'race': race, 'emotion': emotion}

            # Insert data into SQLite database
            cursor.execute("INSERT INTO PeopleData (Age, Gender, Race, Emotion) VALUES (?, ?, ?, ?)", (age, gender, race, emotion))
            conn.commit()

        # Draw rectangle around face and label with predicted attributes
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, face_identities[(x, y, w, h)]['emotion'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    key = cv2.waitKey(1)
    if key == ord('q'): 
        break

# Release the capture
cap.release()

# Close SQLite connection
conn.close()

# Close all OpenCV windows
cv2.destroyAllWindows()
