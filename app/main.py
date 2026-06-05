import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

# then all other imports below
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.head_movement   import router as head_movement_router
from app.api.blink_detection import router as blink_detection_router
from app.api.face_recognition import router as face_recognition_router

app = FastAPI(title="AI Attendance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(head_movement_router)
app.include_router(blink_detection_router)
app.include_router(face_recognition_router)
