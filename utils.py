import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_faces(gray_img):
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    return faces

def preprocess_face(face):
    resized_face = cv2.resize(face, (48, 48))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
    return reshaped_face
