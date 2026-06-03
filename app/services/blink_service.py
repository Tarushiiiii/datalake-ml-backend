import numpy as np

# TEMPORARILY DISABLED
# from tensorflow.keras.models import load_model
#
# eye_model = load_model(
#     "models/eye_classifier.keras/",
#     compile=False,
# )


def detect_blink(frame):
    """
    Placeholder blink detection.
    Real ML integration later.
    """

    prediction = np.random.random()

    return {
        "success": prediction > 0.5,
        "blink_count": 2,
    }