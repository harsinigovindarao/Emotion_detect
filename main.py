import cv2
import numpy as np
from deepface import DeepFace

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame.")
            break

        # Convert frame to grayscale for better face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("🔍 No face detected.")
        
        # Convert to RGB for DeepFace
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (x, y, w, h) in faces:
            face = frame_rgb[y:y+h, x:x+w]  # Crop face region
            print("✅ Face detected! Analyzing...")

            try:
                # Analyze detected face
                result = DeepFace.analyze(face, actions=['emotion', 'age', 'gender'], enforce_detection=False)

                # Extract attributes
                emotion = result[0]['dominant_emotion']
                age = result[0]['age']
                gender = result[0]['dominant_gender']

                # Print result in console
                print(f"😀 Emotion: {emotion} | 🎂 Age: {age} | 🚻 Gender: {gender}")

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display text
                text = f"{emotion} | {age} yrs | {gender}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            except Exception as e:
                print(f"❌ DeepFace Error: {e}")

        # Show output window
        cv2.imshow("🎥 Real-Time Face Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            print("🔴 Stopping webcam...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("🚀 Starting Real-Time Emotion, Age & Gender Detection...")
    detect_faces()
