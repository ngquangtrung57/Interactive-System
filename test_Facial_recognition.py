import sqlite3
from deepface.DeepFace import analyze, verify

# Function to analyze an image and categorize attributes
def analyze_image(img_path):
    return analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'])

# Function to check if a similar person already exists in the database
def find_similar_person(cursor, age, gender, race, emotion):
    cursor.execute('SELECT PersonID FROM PeopleData WHERE Age=? AND Gender=? AND Race=? AND Emotion=?', (age, gender, race, emotion))
    result = cursor.fetchone()
    return result[0] if result else None

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

# Import image paths for different people
image_paths = [
    ("0", "tests/dataset/img5.jpg"),
    ("1", "tests/dataset/img6.jpg"),
    ("2", "tests/dataset/img7.jpg"),
    ("3", "tests/dataset/img8.jpg")
]

# Analyze images and store data for each person
for person, img_path in image_paths:
    attribute = analyze_image(img_path)

    # Extract attributes for the person
    age = attribute[0]['age']
    dominant_gender = attribute[0]['dominant_gender']
    dominant_race = attribute[0]['dominant_race']
    dominant_emotion = attribute[0]['dominant_emotion']

    # Check if a similar person already exists in the database
    existing_person_id = find_similar_person(cursor, age, dominant_gender, dominant_race, dominant_emotion)

    if existing_person_id:
        print(f"Similar person already exists with ID {existing_person_id}")
    else:
        # Insert new record
        cursor.execute('INSERT INTO PeopleData (Age, Gender, Race, Emotion) VALUES (?, ?, ?, ?)',
                       (age, dominant_gender, dominant_race, dominant_emotion))
        print("Inserted new person")

# Commit changes and close connection
conn.commit()
conn.close()
