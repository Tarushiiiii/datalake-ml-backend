import cv2
import mediapipe as mp
import numpy as np

from keras.models import load_model
from keras.layers import Dense


# PATCH Dense

original_dense_init = Dense.__init__


def patched_dense_init(
    self,
    *args,
    **kwargs,
):

    kwargs.pop(
        "quantization_config",
        None,
    )

    original_dense_init(
        self,
        *args,
        **kwargs,
    )


Dense.__init__ = patched_dense_init


# LOAD MODEL

eye_model = load_model(
    "models/blink_eye.keras",
    compile=False,
    safe_mode=False,
)

print("✅ Eye model loaded")


# MEDIAPIPE

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
)


LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]


# EYE CROP

def crop_eye(
    frame,
    eye_points,
):

    x_coords = [p[0] for p in eye_points]
    y_coords = [p[1] for p in eye_points]

    x1 = max(min(x_coords) - 20, 0)
    y1 = max(min(y_coords) - 20, 0)

    x2 = min(max(x_coords) + 20, frame.shape[1])
    y2 = min(max(y_coords) + 20, frame.shape[0])

    return frame[
        y1:y2,
        x1:x2,
    ]


# BLINK DETECTION

def detect_blink(frame):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:

        return {
            "success": False,
            "blink_count": 0,
            "message": "No face detected",
        }

    face = results.multi_face_landmarks[0]

    h, w, _ = frame.shape

    points = []

    for lm in face.landmark:

        points.append(
            (
                int(lm.x * w),
                int(lm.y * h),
            )
        )

    left_eye = [
        points[i]
        for i in LEFT_EYE
    ]

    right_eye = [
        points[i]
        for i in RIGHT_EYE
    ]

    left_crop = crop_eye(
        frame,
        left_eye,
    )

    right_crop = crop_eye(
        frame,
        right_eye,
    )

    if (
        left_crop.size == 0
        or right_crop.size == 0
    ):

        return {
            "success": False,
            "blink_count": 0,
            "message": "Eye crop failed",
        }

    # resize EXACTLY like notebook
    left_crop = cv2.resize(
        left_crop,
        (96, 96),
    )

    right_crop = cv2.resize(
        right_crop,
        (96, 96),
    )

    # normalize
    left_crop = (
        left_crop.astype(np.float32)
        / 255.0
    )

    right_crop = (
        right_crop.astype(np.float32)
        / 255.0
    )

    # batch dimension
    left_crop = np.expand_dims(
        left_crop,
        axis=0,
    )

    right_crop = np.expand_dims(
        right_crop,
        axis=0,
    )

    # prediction
    left_pred = eye_model.predict(
        left_crop,
        verbose=0,
    )[0][0]

    right_pred = eye_model.predict(
        right_crop,
        verbose=0,
    )[0][0]

    prediction = (
        left_pred +
        right_pred
    ) / 2

    print(
        "Blink prediction:",
        prediction,
    )

    success = prediction > 0.5

    return {
        "success": success,
        "blink_count": (
            1 if success else 0
        ),
        "confidence": float(prediction),
    }