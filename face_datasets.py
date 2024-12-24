import cv2
import os

# Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Start the webcam
vid_cam = cv2.VideoCapture(0)

# Face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Prompt user for face ID
face_id = input('Enter User ID: ')

# Image count
count = 0

# Check existence of path
assure_path_exists("training_data/")

while True:
    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle and save the image
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite(f"training_data/Person.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
        cv2.imshow('Capturing Images', image_frame)

    # Stop when 'q' is pressed or count exceeds 100
    if cv2.waitKey(100) & 0xFF == ord('q') or count >= 500:

        break

vid_cam.release()
cv2.destroyAllWindows()
