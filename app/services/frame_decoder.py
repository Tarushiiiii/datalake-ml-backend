import base64
import cv2
import numpy as np


def decode_base64_frame(base64_string: str):
    image_bytes = base64.b64decode(base64_string)

    np_arr = np.frombuffer(image_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return frame