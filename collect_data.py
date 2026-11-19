import cv2, csv, os, time
import mediapipe as mp

DATA_FILE = "Data/landmarks.csv"
SAMPLES_PER_RUN = 100

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

# Count samples already saved
counts = {}
if os.path.exists(DATA_FILE):
    with open(DATA_FILE) as f:
        for row in csv.reader(f):
            if row: counts[row[0]] = counts.get(row[0], 0) + 1

f = open(DATA_FILE, "a", newline="")
writer = csv.writer(f)
cap = cv2.VideoCapture(0)

label, recorded, countdown_end = None, 0, 0

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    hand = res.multi_hand_landmarks[0] if res.multi_hand_landmarks else None
    if hand:
        draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break                                  # Esc = quit

    # Start a new recording
    if label is None:
        if ord('a') <= key <= ord('z'):
            label = chr(key).upper()
        elif key == ord('1'):
            label = "NOTHING"
        if label:
            recorded, countdown_end = 0, time.time() + 3

    # Countdown / recording
    if label:
        remaining = countdown_end - time.time()
        if remaining > 0:
            text = f"{label}: get ready... {int(remaining) + 1}"
        else:
            if hand:
                writer.writerow([label] + [v for p in hand.landmark for v in (p.x, p.y, p.z)])
                recorded += 1
            text = f"Recording {label}: {recorded}/{SAMPLES_PER_RUN}"
            if recorded >= SAMPLES_PER_RUN:
                counts[label] = counts.get(label, 0) + recorded
                f.flush()
                print(f"{label} total samples: {counts[label]}")
                label = None
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Letter key = record | 1 = nothing | Esc = quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Collect Data", frame)

f.close(); cap.release(); cv2.destroyAllWindows()