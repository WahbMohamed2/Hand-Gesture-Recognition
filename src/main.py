import streamlit as st
import cv2
import numpy as np
from model_loader import load_model
from hand_segmenter import segment_hand
from predictor import predict_gesture

st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

st.title("🖐️ Real-Time Hand Gesture Recognition")
st.write(
    "This app uses your webcam to recognize hand gestures and display the contour in real time."
)

# Load model
model = load_model()

# Checkbox to start webcam
run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("No frame detected!")
        break

    frame = cv2.flip(frame, 1)
    segmented, contour = segment_hand(frame)

    if segmented is not None:
        prediction = predict_gesture(model, segmented)
        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()
st.write("Webcam stopped.")
