import cv2
import numpy as np
import os

# Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Create Local Binary Patterns Histograms for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("saved_model/")

# Load the saved pre-trained model
recognizer.read('saved_model/s_model.yml')

# Load prebuilt classifier for Frontal Face detection
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath)

# Font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Confidence threshold for unknown classification
UNKNOWN_THRESHOLD = 50

# Initialize and start the video frame capture from webcam
cam = cv2.VideoCapture(0)

# Looping starts here
while True:
    # Read the video frame
    ret, im = cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Getting all faces from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    # For each face in faces, predict using the pre-trained model
    for (x, y, w, h) in faces:
        # Create rectangle around the face
        cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

        # Recognize the face using the trained model
        Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Set the name according to ID and confidence
        if confidence < UNKNOWN_THRESHOLD:  # Higher confidence means lower error
            if Id == 1:
                name = "Sindhu"
            elif Id == 2:
                name = "Vyshu"
            elif Id == 4:
                name = "Akshaya"
            elif Id == 3:
                name = "Ushodaya"
            elif Id==5:
                pass
            else:
                name = "Unknown"
            Id = f"{name} {120 - confidence:.2f}%"
        else:
            Id = "Unknown"

        # Display the name and rectangle around the face
        cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
        cv2.putText(im, str(Id), (x, y - 40), font, 1, (255, 255, 255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('Face Recognition', im)

    # Press 'q' to close the program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Terminate video
cam.release()

# Close all windows
cv2.destroyAllWindows()

