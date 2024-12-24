import cv2
import numpy as np
from PIL import Image
import os

# Load the trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('saved_model/s_model.yml')
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Path to test images
test_path = 'test_data/'  # Update this if your folder is located elsewhere

# Check if the test path exists
if not os.path.exists(test_path):
    print(f"Error: Test path '{test_path}' does not exist. Please check the path and try again.")
    exit()

# Get all image paths in the test directory
imagePaths = [os.path.join(test_path, f) for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))]

correct = 0
total = 0

# Process each image in the test directory
for imagePath in imagePaths:
    try:
        # Read test image and convert to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        # Extract ID from the file name
        try:
            actual_id = int(os.path.split(imagePath)[-1].split(".")[1])
        except (IndexError, ValueError):
            print(f"Error: Unable to extract ID from file name '{imagePath}'. Skipping this image.")
            continue

        # Detect face in the image
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            total += 1
            predicted_id, confidence = recognizer.predict(img_numpy[y:y+h, x:x+w])
            print(f"Actual ID: {actual_id}, Predicted ID: {predicted_id}, Confidence: {confidence:.2f}")

            # Count as correct if IDs match
            if actual_id == predicted_id:
                correct += 1

    except Exception as e:
        print(f"Error processing image '{imagePath}': {e}")

# Calculate and print accuracy
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"Model Accuracy: {accuracy:.2f}%")
