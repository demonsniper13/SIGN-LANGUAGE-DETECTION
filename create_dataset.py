import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

if not os.path.exists(DATA_DIR):
    print("❌ Error: Data directory not found. Run collect_imgs.py first.")
    exit()

data = []
labels = []
skipped_images = 0

print("🔍 Processing images and extracting hand landmarks...")

for dir_ in sorted(os.listdir(DATA_DIR)):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    # Skip non-directory items
    if not os.path.isdir(dir_path):
        continue
    
    print(f"\n📂 Processing class {dir_}...")
    processed_count = 0
    
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        
        # Read image
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"⚠️ Could not read {img_path}")
            skipped_images += 1
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []
                
                # Collect all x, y coordinates
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                
                # Normalize coordinates relative to bounding box
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))
                
                # Only add if we have exactly 42 features (21 landmarks * 2 coords)
                if len(data_aux) == 42:
                    data.append(data_aux)
                    labels.append(dir_)
                    processed_count += 1
        else:
            skipped_images += 1
    
    print(f"   ✅ Processed {processed_count} images")

print(f"\n📊 Summary:")
print(f"   Total samples: {len(data)}")
print(f"   Skipped images: {skipped_images}")

if len(data) == 0:
    print("❌ Error: No valid hand landmarks detected. Please recollect data.")
    exit()

# Save dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Dataset saved to data.pickle")