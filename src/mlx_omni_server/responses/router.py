"""
FastAPI router for /v1/responses endpoint.

Provides OpenAI Responses API compatible endpoints for
multi-turn conversations with extended capabilities.

Contributed by LibraxisAI - https://libraxis.ai
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..utils.logger import logger
from .adapter import ResponsesAdapter
from .schema import ResponseRequest, ResponseResponse, build_error_response

router = APIRouter(tags=["responses"])

# Shared adapter instance
_adapter: ResponsesAdapter | None = None


def get_adapter() -> ResponsesAdapter:
    """Get or create the shared ResponsesAdapter."""
    global _adapter
    if _adapter is None:
        _adapter = ResponsesAdapter()
    return _adapter


@router.post("/v1/responses", response_model=ResponseResponse)
async def create_response(
    request: Request,
    body: ResponseRequest,
) -> JSONResponse | StreamingResponse:
    """
    Create a response using the Responses API.

    This endpoint provides compatibility with OpenAI's Responses API,
    supporting multi-turn conversations with extended capabilities:

    - Text generation with reasoning
    - Multimodal input (images, audio, video)
    - Tool/function calling
    - Structured output (JSON schema)
    - Streaming responses

    The endpoint automatically routes requests to the appropriate
    backend based on content type and model configuration.
    """
    adapter = get_adapter()

    if body.stream:
        # Streaming response
        async def event_generator():
            try:
                async for event in adapter.generate_stream(body):
                    event_type = event.get("type", "")
                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(event)}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                error_event = {
                    "type": "error",
                    "error": {
                        "message": str(e),
                        "code": "internal_error",
                    },
                }
                yield f"event: error\n"
                yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming response
        try:
            response = await adapter.generate(body)
            return JSONResponse(content=response.model_dump(exclude_none=True))
        except Exception as e:
            logger.error(f"Response generation error: {e}", exc_info=True)
            error_response = build_error_response(
                str(e),
                error_code="internal_error",
                model=body.model,
            )
            return JSONResponse(
                content=error_response.model_dump(exclude_none=True),
                status_code=500,
            )


@router.get("/v1/responses/{response_id}")
async def get_response(response_id: str) -> JSONResponse:
    """
    Retrieve a stored response by ID.

    Note: Response storage is not yet implemented.
    This endpoint is provided for API compatibility.
    """
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"Response '{response_id}' not found",
                "type": "invalid_request_error",
                "code": "response_not_found",
            }
        },
    )


@router.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str) -> JSONResponse:
    """
    Delete a stored response.

    Note: Response storage is not yet implemented.
    This endpoint is provided for API compatibility.
    """
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"Response '{response_id}' not found",
                "type": "invalid_request_error",
                "code": "response_not_found",
            }
        },
    )


@router.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str) -> JSONResponse:
    """
    Cancel an in-progress response.

    Note: Background response processing is not yet implemented.
    This endpoint is provided for API compatibility.
    """
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"Response '{response_id}' not found or not cancellable",
                "type": "invalid_request_error",
                "code": "response_not_found",
            }
        },
    )


@router.get("/v1/responses/{response_id}/input_items")
async def list_input_items(response_id: str) -> JSONResponse:
    """
    List input items for a response.

    Note: Response storage is not yet implemented.
    This endpoint is provided for API compatibility.
    """
    raise HTTPException(
        status_code=404,
        detail={
            "error": {
                "message": f"Response '{response_id}' not found",
                "type": "invalid_request_error",
                "code": "response_not_found",
            }
        },
    )
