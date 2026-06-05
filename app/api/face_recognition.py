from fastapi import APIRouter

from app.schemas.requests import (
    FrameRequest,
)

from app.schemas.responses import (
    FaceRecognitionResponse,
)

from app.services.frame_decoder import (
    decode_base64_frame,
)

from app.services.face_service import (
    verify_face,
)


router = APIRouter(
    tags=["Face Recognition"]
)


@router.post(
    "/face-recognition",
    response_model=
        FaceRecognitionResponse,
)
async def face_recognition(
    payload: FrameRequest,
):

    frame = decode_base64_frame(
        payload.frame
    )

    result = verify_face(frame)

    return FaceRecognitionResponse(**result)
