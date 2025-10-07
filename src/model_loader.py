from tensorflow.keras.models import load_model as keras_load_model
import os


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "hand_gesture_recognition.h5")
    model = keras_load_model(model_path)
    return model
