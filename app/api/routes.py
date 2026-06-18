from typing import Callable

from fastapi import APIRouter, HTTPException, Request

from app.core.config import Settings, get_settings
from app.core.normalizer import PayloadLimitError
from app.core.validator import ContentValidator
from app.models.schemas import StoryPayload, ValidationResponse

router = APIRouter()

ValidatorFactory = Callable[[Settings], ContentValidator]


@router.post("/validate", response_model=ValidationResponse)
async def validate(payload: StoryPayload, request: Request) -> ValidationResponse:
    settings = get_settings()
    factory: ValidatorFactory | None = getattr(request.app.state, "validator_factory", None)
    validator = factory(settings) if factory else ContentValidator(settings)

    try:
        return await validator.validate(payload)
    except PayloadLimitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
