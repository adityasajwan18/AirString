import mediapipe as mp
try:
    print(mp.solutions)
    print("SUCCESS: MediaPipe is working!")
except AttributeError as e:
    print(f"FAIL: {e}")