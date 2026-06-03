from fastapi import APIRouter

from app.schemas.requests import FrameRequest
from app.schemas.responses import BlinkDetectionResponse

from app.services.frame_decoder import decode_base64_frame
from app.services.blink_service import detect_blink

router = APIRouter()

@router.post(
    "/blink-detection",
    response_model=BlinkDetectionResponse,
)
async def blink_detection(
    payload: FrameRequest,
):
    frame = decode_base64_frame(payload.frame)

    result = detect_blink(frame)

    return result