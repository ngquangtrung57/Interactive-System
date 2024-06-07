import cv2
from deepface import DeepFace

DeepFace.stream(db_path = "/home/interactivesystem/Desktop/Facial-Emotion-Recognition-using-OpenCV-and-Deepface/deepface/tests/dataset",
                model_name = "VGG-Face",   #Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet
                detector_backend = "yolov8",    #Options: 'opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'
                time_threshold = 10,    # time that the frame will last for after detect a face
                frame_threshold = 5,    # time needed to detect faces
)
