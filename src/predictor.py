import numpy as np


def predict_gesture(model, image):
    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    return f"Class {class_idx}"
