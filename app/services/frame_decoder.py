import base64
import cv2
import numpy as np


def decode_base64_frame(base64_string: str):
    # Strip data-URL prefix if present (e.g. "data:image/jpeg;base64,...")
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    image_bytes = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("cv2.imdecode returned None — invalid image bytes")

    return frame