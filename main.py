import cv2
import mediapipe as mp
import pygame
import time

# --- 1. Audio Setup ---
pygame.mixer.init()
# Load your guitar chord sounds here (Ensure these files exist in your folder)
# If you don't have files yet, the code will run but just print the chord name.
sounds = {}
try:
    sounds = {
        'C': pygame.mixer.Sound('C.wav'),
        'G': pygame.mixer.Sound('G.wav'),
        'D': pygame.mixer.Sound('D.wav'),
        # Add more as needed
    }
except FileNotFoundError:
    print("WARNING: Audio files (C.wav, G.wav, D.wav) not found. Audio will be skipped.")

# --- 2. MediaPipe Hand Setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- 3. Finger Counting Logic ---
# Tip IDs: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
tip_ids = [4, 8, 12, 16, 20]

def count_fingers(lm_list):
    """Returns the number of fingers that are 'up'."""
    fingers = []

    # Thumb (Side logic: checks if tip is to the right/left of the knuckle depending on hand)
    # Assuming right hand facing camera:
    if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 Fingers (Height logic: checks if tip is above the middle knuckle)
    for id in range(1, 5):
        if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# --- 4. Main Loop ---
cap = cv2.VideoCapture(0)
last_chord = None

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip image for mirror view
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process hands
    results = hands.process(img_rgb)
    
    current_chord = "None"
    
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Draw skeleton
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # Get landmark coordinates
            lm_list = []
            h, w, c = img.shape
            for id, lm in enumerate(hand_lms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            if len(lm_list) != 0:
                fingers = count_fingers(lm_list)
                total_fingers = fingers.count(1)
                
                # --- CHORD MAPPING RULES ---
                # 2 Fingers (Peace Sign) -> G Major
                if fingers[1] == 1 and fingers[2] == 1 and total_fingers == 2:
                    current_chord = "G Major"
                    key = 'G'
                
                # 5 Fingers (Open Palm) -> D Major
                elif total_fingers == 5:
                    current_chord = "D Major"
                    key = 'D'
                
                # Thumb only (Thumbs up) -> C Major
                elif fingers[0] == 1 and total_fingers == 1:
                    current_chord = "C Major"
                    key = 'C'
                
                else:
                    current_chord = "Mute"
                    key = None

                # --- TRIGGER AUDIO ---
                # Only play if the chord changed to avoid spamming sound every frame
                if current_chord != last_chord:
                    if key and key in sounds:
                        sounds[key].play()
                    last_chord = current_chord

    # --- UI Display ---
    # Draw a rectangle and text
    cv2.rectangle(img, (20, 20), (300, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, f'Chord: {current_chord}', (30, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("AirString - Virtual Guitar", img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()