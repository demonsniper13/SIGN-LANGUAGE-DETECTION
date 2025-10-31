import pickle
import cv2
import mediapipe as mp
import numpy as np
import os

# Check if model exists
if not os.path.exists('./model.p'):
    print("❌ Error: model.p not found. Run train_classifier.py first.")
    exit()

# Load trained model
print("📂 Loading trained model...")
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
print("✅ Model loaded successfully!")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Unable to access camera.")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, 
                       min_tracking_confidence=0.5)

# Gesture labels (A-Z)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

print("\n🎥 Starting real-time sign language detection...")
print("   Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Unable to capture frame. Check camera connection.")
        break

    # Mirror the camera feed
    frame = cv2.flip(frame, 1)

    # Process frame
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on mirrored frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmark data
            data_aux = []
            x_ = []
            y_ = []

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            # Only predict if we have the correct number of features
            if len(data_aux) == 42:
                # Bounding box around hand
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                # Ensure bounding box is within frame
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)

                try:
                    # Prediction
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Display results
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add background for text
                    text_size = cv2.getTextSize(predicted_character, 
                                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                    cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), 
                                 (x1 + text_size[0], y1), (0, 255, 0), -1)
                    
                    cv2.putText(frame, predicted_character, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
                except Exception as e:
                    print(f"⚠️ Prediction error: {e}")

    # Show instructions
    cv2.putText(frame, 'Press Q to quit', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Show mirrored frame
    cv2.imshow('Sign Language Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Detection stopped. Goodbye!")