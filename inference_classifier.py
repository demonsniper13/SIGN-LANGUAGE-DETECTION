import pickle
import cv2
import mediapipe as mp
import numpy as np
import os
import pyttsx3
import subprocess
import time
import threading

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
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'hello', 6: 'G', 7: 'H', 8: 'I',
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Initialize Text-to-Speech engine
print("🔊 Initializing text-to-speech engine...")
try:
    tts_engine = pyttsx3.init()
    # Set speech rate (words per minute)
    tts_engine.setProperty('rate', 150)
    # Set volume (0.0 to 1.0)
    tts_engine.setProperty('volume', 0.9)
    print("✅ Text-to-speech engine ready!")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize TTS engine: {e}")
    tts_engine = None

# TTS variables for word completion
last_spoken_word = ""  # Track last spoken word to avoid repeats

# Sentence formation variables
current_sentence = ""  # The sentence being built
current_sign = None  # Currently detected sign
sign_start_time = None  # When the current sign started being held
sign_hold_duration = 1.0  # Seconds to hold a sign before adding to sentence
last_sign_added = None  # Last sign that was added to sentence
pause_start_time = None  # When no sign was detected (pause started)
pause_duration = 2.0  # Seconds of pause before adding a space

# App trigger configuration
GOOGLE_CLASSROOM_URL = "https://classroom.google.com"
GOOGLE_CLASSROOM_TRIGGER = 'G'  # Hold 'G' for 5 seconds to open Google Classroom
BRAVE_TRIGGER = 'B'  # Hold 'B' for 5 seconds to open Brave browser
trigger_hold_duration = 5.0  # Seconds to hold trigger symbol before opening app
trigger_start_time = None  # When trigger symbol started being held
current_trigger_symbol = None  # Currently detected trigger symbol
last_trigger_time = 0
trigger_cooldown = 3.0  # seconds between triggers to prevent multiple opens

def speak_text(text):
    """Speak text using TTS engine in a separate thread"""
    if tts_engine is None:
        return
    try:
        def speak():
            tts_engine.say(text)
            tts_engine.runAndWait()
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()
    except Exception as e:
        print(f"⚠️ TTS error: {e}")

def open_google_classroom():
    """Open Google Classroom in Chrome or Brave browser"""
    global last_trigger_time
    current_time = time.time()
    
    # Prevent multiple triggers in quick succession
    if current_time - last_trigger_time < trigger_cooldown:
        return
    
    last_trigger_time = current_time
    
    # Try to find and open Chrome or Brave
    browsers = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
    ]
    
    browser_found = False
    for browser_path in browsers:
        if os.path.exists(browser_path):
            try:
                subprocess.Popen([browser_path, GOOGLE_CLASSROOM_URL])
                print(f"✅ Opening Google Classroom in {os.path.basename(browser_path)}")
                browser_found = True
                break
            except Exception as e:
                print(f"⚠️ Error opening browser: {e}")
    
    if not browser_found:
        # Fallback: use default browser
        try:
            import webbrowser
            webbrowser.open(GOOGLE_CLASSROOM_URL)
            print("✅ Opening Google Classroom in default browser")
        except Exception as e:
            print(f"⚠️ Error opening browser: {e}")

def open_brave_browser():
    """Open Brave browser"""
    global last_trigger_time
    current_time = time.time()
    
    # Prevent multiple triggers in quick succession
    if current_time - last_trigger_time < trigger_cooldown:
        return
    
    last_trigger_time = current_time
    
    # Try to find and open Brave browser - check multiple possible paths
    # User's specific path (shortcut) - highest priority
    brave_paths = [
        # Shortcut in Start Menu (user-specific)
        os.path.join(os.path.expanduser("~"), r"AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Brave.lnk"),
        # Direct executable paths
        # r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        # r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        # os.path.join(os.path.expanduser("~"), r"AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe"),
        # os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        # os.path.expandvars(r"%PROGRAMFILES%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        # os.path.expandvars(r"%PROGRAMFILES(X86)%\BraveSoftware\Brave-Browser\Application\brave.exe"),
    ]
    
    browser_found = False
    for browser_path in brave_paths:
        # Paths are already expanded, just check if they exist
        if os.path.exists(browser_path):
            try:
                # Check if it's a .lnk file (shortcut)
                if browser_path.lower().endswith('.lnk'):
                    # For .lnk files, use shell=True to let Windows handle the shortcut
                    subprocess.Popen(f'"{browser_path}"', shell=True)
                    print(f"✅ Opening Brave browser from shortcut: {browser_path}")
                    browser_found = True
                    break
                else:
                    # For .exe files, try opening directly
                    subprocess.Popen([browser_path], shell=False)
                    print(f"✅ Opening Brave browser from: {browser_path}")
                    browser_found = True
                    break
            except Exception as e:
                print(f"⚠️ Error opening Brave browser: {e}")
                # Try with shell=True as fallback
                try:
                    subprocess.Popen(f'"{browser_path}"', shell=True)
                    print(f"✅ Opening Brave browser (shell method)")
                    browser_found = True
                    break
                except Exception as e2:
                    print(f"⚠️ Error with shell method: {e2}")
    
    # Try using Windows start command
    if not browser_found:
        try:
            # Try to find Brave in common locations using where command
            result = subprocess.run(['where', 'brave.exe'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                brave_path = result.stdout.strip().split('\n')[0]
                subprocess.Popen([brave_path], shell=False)
                print(f"✅ Opening Brave browser (found via where command)")
                browser_found = True
        except Exception as e:
            pass  # where command might not be available
    
    # Try using webbrowser module with Brave
    if not browser_found:
        try:
            import webbrowser
            # Try to register Brave browser
            brave_path = None
            for path in brave_paths:
                if os.path.exists(path):
                    brave_path = path
                    break
            
            if brave_path:
                # Register Brave with webbrowser
                webbrowser.register('brave', None, 
                                  webbrowser.BackgroundBrowser(brave_path))
                webbrowser.get('brave').open('about:blank')
                print(f"✅ Opening Brave browser (via webbrowser module)")
                browser_found = True
        except Exception as e:
            print(f"⚠️ Error with webbrowser method: {e}")
    
    # Final fallback: try using start command with brave
    if not browser_found:
        try:
            subprocess.Popen(['start', 'brave'], shell=True)
            print(f"✅ Attempting to open Brave browser (start command)")
            browser_found = True
        except Exception as e:
            pass
    
    if not browser_found:
        print("⚠️ Brave browser not found. Please install Brave or check installation path.")
        print("   Common installation paths:")
        print("   - C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe")
        print("   - %LOCALAPPDATA%\\BraveSoftware\\Brave-Browser\\Application\\brave.exe")

print("\n🎥 Starting real-time sign language detection...")
print("   Press 'Q' to quit")
print("   Press 'C' to clear sentence")
print(f"   Hold '{GOOGLE_CLASSROOM_TRIGGER}' for 5 seconds to open Google Classroom")
print(f"   Hold '{BRAVE_TRIGGER}' for 5 seconds to open Brave browser")
print("   Words will be spoken when completed (space added)")
print("   Hold a sign for 1 second to add to sentence")
print("   2 second pause adds a space and speaks the word")

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
    
    current_time = time.time()
    sign_detected = False

    if results.multi_hand_landmarks:
        sign_detected = True
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
                    
                    # Sentence formation: Track sign holding (skip if it's a trigger symbol)
                    # Note: TTS will speak when word is completed (space encountered)
                    if predicted_character not in [GOOGLE_CLASSROOM_TRIGGER, BRAVE_TRIGGER]:
                        if current_sign != predicted_character:
                            # New sign detected, reset timer
                            current_sign = predicted_character
                            sign_start_time = current_time
                            pause_start_time = None  # Reset pause timer
                        elif sign_start_time is not None:
                            # Same sign being held, check if held long enough
                            hold_time = current_time - sign_start_time
                            if hold_time >= sign_hold_duration and last_sign_added != predicted_character:
                                # Add sign to sentence
                                current_sentence += predicted_character
                                last_sign_added = predicted_character
                                print(f"📝 Sentence: {current_sentence}")
                                # Visual feedback
                                cv2.putText(frame, 'Added!', (x1, y1 - 40),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # Check for trigger symbols (G for Google Classroom, B for Brave)
                    if predicted_character == GOOGLE_CLASSROOM_TRIGGER or predicted_character == BRAVE_TRIGGER:
                        if current_trigger_symbol != predicted_character:
                            # New trigger symbol detected, start tracking
                            current_trigger_symbol = predicted_character
                            trigger_start_time = current_time
                            # Reset sentence tracking when trigger is active
                            current_sign = None
                            sign_start_time = None
                        elif trigger_start_time is not None:
                            # Same trigger symbol being held, check if held long enough
                            hold_time = current_time - trigger_start_time
                            if hold_time >= trigger_hold_duration:
                                # Trigger activated!
                                if predicted_character == GOOGLE_CLASSROOM_TRIGGER:
                                    open_google_classroom()
                                    cv2.putText(frame, 'Opening Google Classroom!', (10, 90),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                                elif predicted_character == BRAVE_TRIGGER:
                                    open_brave_browser()
                                    cv2.putText(frame, 'Opening Brave Browser!', (10, 90),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                                # Reset trigger tracking
                                current_trigger_symbol = None
                                trigger_start_time = None
                    else:
                        # Not a trigger symbol, reset trigger tracking
                        if current_trigger_symbol is not None:
                            current_trigger_symbol = None
                            trigger_start_time = None
                        
                except Exception as e:
                    print(f"⚠️ Prediction error: {e}")
    
    # Handle pause (no sign detected) for adding spaces
    if not sign_detected:
        # Reset trigger tracking when no sign is detected
        if current_trigger_symbol is not None:
            current_trigger_symbol = None
            trigger_start_time = None
        
        if pause_start_time is None:
            # Start tracking pause
            pause_start_time = current_time
            # Reset sign tracking
            current_sign = None
            sign_start_time = None
            last_sign_added = None
        else:
            # Check if pause is long enough to add a space
            pause_time = current_time - pause_start_time
            if pause_time >= pause_duration and current_sentence and current_sentence[-1] != ' ':
                # Extract the last completed word (before the space we're about to add)
                words = current_sentence.split()
                if words:  # If there are words in the sentence
                    last_word = words[-1]  # Get the last word
                    # Only speak if it's a new word (not already spoken)
                    if last_word != last_spoken_word:
                        speak_text(last_word)
                        last_spoken_word = last_word
                        print(f"🔊 Spoke word: {last_word}")
                
                current_sentence += ' '
                print(f"📝 Sentence: {current_sentence}")
                pause_start_time = None  # Reset to prevent multiple spaces

    # Show instructions
    cv2.putText(frame, 'Press Q to quit | C to clear', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hold '{GOOGLE_CLASSROOM_TRIGGER}' 5s: Google Classroom", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Hold '{BRAVE_TRIGGER}' 5s: Brave Browser", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Display current sentence
    if current_sentence:
        # Calculate text size for background
        sentence_text = f"Sentence: {current_sentence}"
        text_size = cv2.getTextSize(sentence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        # Draw background rectangle
        cv2.rectangle(frame, (10, H - 50), (20 + text_size[0], H - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, H - 50), (20 + text_size[0], H - 10), (255, 255, 255), 2)
        # Draw sentence text
        cv2.putText(frame, sentence_text, (15, H - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show progress indicator for trigger symbols
    if sign_detected and current_trigger_symbol and trigger_start_time is not None:
        hold_time = current_time - trigger_start_time
        progress = min(hold_time / trigger_hold_duration, 1.0)
        # Draw progress bar (red for trigger)
        bar_width = 200
        bar_height = 25
        bar_x = W - bar_width - 10
        bar_y = 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 0, 255), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        trigger_name = "Google Classroom" if current_trigger_symbol == GOOGLE_CLASSROOM_TRIGGER else "Brave Browser"
        cv2.putText(frame, f'Trigger: {current_trigger_symbol} ({trigger_name})', (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # Show countdown
        remaining = max(0, trigger_hold_duration - hold_time)
        cv2.putText(frame, f'{remaining:.1f}s', (bar_x + bar_width + 10, bar_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Show progress indicator for current sign (only if not a trigger)
    elif sign_detected and current_sign and sign_start_time is not None and current_sign not in [GOOGLE_CLASSROOM_TRIGGER, BRAVE_TRIGGER]:
        hold_time = current_time - sign_start_time
        progress = min(hold_time / sign_hold_duration, 1.0)
        # Draw progress bar
        bar_width = 200
        bar_height = 20
        bar_x = W - bar_width - 10
        bar_y = 30
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        cv2.putText(frame, f'Hold: {current_sign}', (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Show pause indicator
    if not sign_detected and pause_start_time is not None:
        pause_time = current_time - pause_start_time
        if pause_time < pause_duration:
            progress = pause_time / pause_duration
            # Draw pause progress bar
            bar_width = 200
            bar_height = 20
            bar_x = W - bar_width - 10
            bar_y = 60
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (255, 165, 0), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            cv2.putText(frame, 'Pause (space)', (bar_x, bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Show mirrored frame
    cv2.imshow('Sign Language Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') or key == ord('C'):
        # Clear sentence
        current_sentence = ""
        current_sign = None
        sign_start_time = None
        pause_start_time = None
        last_sign_added = None
        current_trigger_symbol = None
        trigger_start_time = None
        last_spoken_word = ""  # Reset spoken word tracking
        print("🗑️ Sentence cleared!")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Detection stopped. Goodbye!")