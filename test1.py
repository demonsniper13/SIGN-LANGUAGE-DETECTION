import cv2
import os

# ROI coordinates
ROI_top, ROI_bottom = 100, 400
ROI_right, ROI_left = 250, 550

# Folder to save dataset images
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)
img_count = 0  # counter for saved images

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Draw ROI
    cv2.rectangle(frame, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 2)
    
    # Extract ROI
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Threshold
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Show
    cv2.imshow("Thresholded", thresh)
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Save ROI when 'c' is pressed
    if key == ord('c'):
        img_count += 1
        file_path = os.path.join(dataset_path, f"roi_{img_count}.jpg")
        cv2.imwrite(file_path, thresh)
        print(f"Image saved: {file_path}")
    
    # Quit when 'q' is pressed
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
