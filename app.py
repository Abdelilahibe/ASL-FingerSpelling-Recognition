import cv2, joblib, numpy as np
import mediapipe as mp
from collections import deque
from utils import normalize

model = joblib.load("Models/asl_model.pkl")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils
FONT = cv2.FONT_HERSHEY_SIMPLEX

STABLE_FRAMES = 15        # frames a letter must be held before it's added
CONF_THRESHOLD = 0.70     # minimum confidence to count a prediction
history = deque(maxlen=STABLE_FRAMES)
message, last_added = "", None

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    letter, conf = "-", 0.0
    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        row = [v for p in hand.landmark for v in (p.x, p.y, p.z)]
        probs = model.predict_proba([normalize(row)])[0]
        i = np.argmax(probs)
        letter, conf = model.classes_[i], probs[i]
        history.append(letter if conf >= CONF_THRESHOLD else None)
    else:
        history.append(None)

    # Stable prediction -> add to message
    if len(history) == STABLE_FRAMES and len(set(history)) == 1:
        s = history[0]
        if s in (None, "NOTHING"):
            last_added = None              # relax or drop hand to allow a repeat letter
        elif s != last_added:
            message += s
            last_added = s

    # UI
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.putText(frame, f"Letter: {letter}  {conf*100:.0f}%", (10, 45), FONT, 1.1, (0, 255, 0), 2)
    cv2.rectangle(frame, (w - 220, 25), (w - 20, 45), (80, 80, 80), -1)
    cv2.rectangle(frame, (w - 220, 25), (w - 220 + int(200 * conf), 45), (0, 255, 0), -1)
    cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, "Message: " + message[-30:], (10, h - 20), FONT, 1, (255, 255, 255), 2)
    cv2.imshow("SignFrame AI", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break                      # Esc = quit
    elif key == 32: message += " "           # Space = add a space
    elif key == 8: message = message[:-1]    # Backspace = delete last letter
    elif key == ord('0'): message = ""       # 0 = clear the message

cap.release(); cv2.destroyAllWindows()