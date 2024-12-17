import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained gender recognition model
gender_model = load_model("fine_tuned_gender_recognition_model.h5")  # Update with your model's path

# Gender labels
gender_labels = ['Male', 'Female']

# Input image size for gender model
gender_img_size = (128, 128)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Preprocess for gender model
        try:
            face_resized = cv2.resize(face_roi, gender_img_size)
            face_normalized = face_resized / 255.0  # Normalize pixel values
            face_input = np.expand_dims(face_normalized, axis=0)

            # Predict gender
            gender_prediction = gender_model.predict(face_input, verbose=0)
            gender_label = gender_labels[np.argmax(gender_prediction)]

            # Display prediction on frame
            label = f"{gender_label} ({gender_prediction[0][np.argmax(gender_prediction)]:.2f})"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error processing face: {e}")

    # Display the video feed with results
    cv2.imshow("Live Gender Detection", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
