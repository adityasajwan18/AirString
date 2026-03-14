# 1_collect_data.py
import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

CHORDS = ["Am", "C", "D", "Em", "G"]
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

print("Chords to collect:", CHORDS)
chord = input("Enter chord name to record (e.g. Am): ").strip()
assert chord in CHORDS, f"Unknown chord. Choose from {CHORDS}"

output_file = os.path.join(DATA_DIR, f"{chord}.csv")
collected = 0
TARGET = 300  # samples per chord

print(f"Recording '{chord}'. Press 's' to save a frame, 'q' to quit.")

with open(output_file, "a", newline="") as f:
    writer = csv.writer(f)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])  # 63 values

                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    writer.writerow([chord] + landmarks)
                    collected += 1
                    print(f"Saved sample {collected}/{TARGET}")
        else:
            cv2.waitKey(1)

        cv2.putText(frame, f"Chord: {chord} | Samples: {collected}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or collected >= TARGET:
            break

cap.release()
cv2.destroyAllWindows()
print(f"Done. {collected} samples saved to {output_file}")