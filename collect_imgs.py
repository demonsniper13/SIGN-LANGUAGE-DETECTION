import os
import cv2

# Directory setup
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26       # A-Z
dataset_size = 100           # images per class

# Try opening default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open camera.")
    exit()

# Set camera properties for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Loop through all gesture classes
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'\n📸 Collecting data for class {j} (Letter: {chr(65+j)})')
    print("👉 Position your hand and press 'Q' when ready to start capturing.")

    # Wait for user to start
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Unable to read from camera.")
            break

        frame = cv2.flip(frame, 1)  # Mirror the camera feed
        cv2.putText(frame, f'Class {j} ({chr(65+j)}) - Ready? Press "Q"', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    # Start capturing dataset images
    counter = 0
    print(f"✅ Starting capture... ({dataset_size} images)")
    
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not captured.")
            continue

        frame = cv2.flip(frame, 1)  # Mirror feed for consistency
        
        # Show progress
        progress_text = f'Capturing: {counter}/{dataset_size}'
        cv2.putText(frame, progress_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)
        cv2.waitKey(25)

        # Save image
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1
    
    print(f"✅ Class {j} complete! ({dataset_size} images saved)")

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Data collection complete!")