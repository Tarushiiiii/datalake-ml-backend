import random


def detect_head_movement(frame):
    return {
        "success": random.choice([True, False]),
        "message": "Head movement detected",
    }