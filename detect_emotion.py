import cv2
import numpy as np
from keras.models import load_model
from utils import detect_faces, preprocess_face, emotion_labels

# Load model
model = load_model('models/emotion_model.h5')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        preprocessed = preprocess_face(roi_gray)
        prediction = model.predict(preprocessed)
        emotion_idx = int(np.argmax(prediction))
        emotion = emotion_labels[emotion_idx]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 255), 2)

    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
