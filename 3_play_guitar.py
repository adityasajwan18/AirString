# 3_play_guitar.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pygame
import os
import time

# ── Setup ──────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("models/chord_model.h5")
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

pygame.mixer.init()
AUDIO_DIR = "audio"

def load_sounds():
    sounds = {}
    for chord in le.classes_:
        path = os.path.join(AUDIO_DIR, f"{chord}.wav")
        if os.path.exists(path):
            sounds[chord] = pygame.mixer.Sound(path)
        else:
            print(f"Warning: No audio file found for chord '{chord}' at {path}")
    return sounds

sounds = load_sounds()

# ── Real-time loop ──────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
CONFIDENCE_THRESHOLD = 0.85
DEBOUNCE_TIME = 0.8  # seconds between chord plays

last_played = ""
last_time = 0

print("Starting guitar hand sign player. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    predicted_chord = None
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            X = np.array(landmarks).reshape(1, -1)
            preds = model.predict(X, verbose=0)[0]
            confidence = float(np.max(preds))
            class_idx = np.argmax(preds)
            predicted_chord = le.inverse_transform([class_idx])[0]

            now = time.time()
            if (confidence >= CONFIDENCE_THRESHOLD and
                    predicted_chord != last_played or
                    now - last_time > DEBOUNCE_TIME):
                if predicted_chord in sounds:
                    sounds[predicted_chord].play()
                    last_played = predicted_chord
                    last_time = now

    # ── Overlay UI ─────────────────────────────────────────────────────
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    chord_text = predicted_chord if predicted_chord else "---"
    conf_text = f"{confidence:.0%}" if predicted_chord else ""
    color = (0, 255, 120) if confidence >= CONFIDENCE_THRESHOLD else (0, 180, 255)

    cv2.putText(frame, f"Chord: {chord_text}", (20, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    cv2.putText(frame, f"Confidence: {conf_text}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    cv2.imshow("Guitar Hand Sign Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()