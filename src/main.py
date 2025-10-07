import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load gesture recognizer model
model = load_model("mp_hand_gesture")

# Load class names
with open("gesture.names", "r") as f:
    classNames = f.read().split("\n")
print("Loaded classes:", classNames)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x, y, c = frame.shape

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for MediaPipe
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)
    className = ""

    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * y)
                lmy = int(lm.y * x)
                landmarks.append([lmx, lmy])

            # Predict gesture
            prediction = model.predict(
                [np.asarray(landmarks).flatten().reshape(1, -1)], verbose=0
            )
            classID = np.argmax(prediction)
            className = classNames[classID]

            # Draw landmarks
            mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)

            # Get bounding box for hand
            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # Draw rectangle around hand
            cv2.rectangle(
                frame,
                (x_min - 20, y_min - 20),
                (x_max + 20, y_max + 20),
                (0, 255, 0),
                2,
            )

            # Display gesture name
            cv2.putText(
                frame,
                className,
                (x_min - 10, y_min - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    # Show final output
    cv2.imshow("Output", frame)

    # Break when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
