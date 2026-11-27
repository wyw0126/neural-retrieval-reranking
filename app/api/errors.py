"""Uniform HTTP errors and request correlation for the retrieval API."""
import logging
from http import HTTPStatus
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Scope, Receive, Send, Message

from app.schemas import ErrorResponse

logger = logging.getLogger("app.api")


def error_response(request: Request, status: int, code: str, message: str,
                   *, details=None, headers=None) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or uuid4().hex
    payload = ErrorResponse(code=code, message=message, request_id=request_id,
                            details=details)
    return JSONResponse(
        status_code=status,
        content=payload.model_dump(exclude_none=True),
        headers=headers,
    )


async def validation_handler(request: Request, exc: RequestValidationError):
    # Do not return raw input, context, or the request body.
    details = [{"location": list(item["loc"]), "type": item["type"]}
               for item in exc.errors()]
    return error_response(request, 422, "VALIDATION_ERROR",
                          "Request validation failed", details=details)


async def http_handler(request: Request, exc: HTTPException):
    codes = {400: "BAD_REQUEST", 401: "UNAUTHORIZED", 403: "FORBIDDEN",
             404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED",
             422: "VALIDATION_ERROR", 429: "TOO_MANY_REQUESTS",
             503: "SERVICE_UNAVAILABLE"}
    try:
        phrase = HTTPStatus(exc.status_code).phrase
    except ValueError:
        phrase = "HTTP error"
    # Only expose deliberately public client errors, not arbitrary server details.
    message = exc.detail if exc.status_code < 500 and isinstance(exc.detail, str) else phrase
    if exc.status_code == 503:
        message = "Service unavailable"
    return error_response(request, exc.status_code,
                          codes.get(exc.status_code, "HTTP_ERROR"), message,
                          headers=exc.headers)


class RequestContextMiddleware:
    """Pure ASGI middleware: correlate all HTTP responses, including failures."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = perf_counter()
        status = 500
        response_started = False

        async def send_with_context(message: Message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["X-Process-Time-Ms"] = f"{(perf_counter()-started)*1000:.3f}"
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        except Exception as exc:
            # Avoid logging raw exception messages or validation payloads.
            logger.error("request_id=%s internal_error=%s", request_id,
                         type(exc).__name__)
            if response_started:
                raise
            response = error_response(Request(scope), 500, "INTERNAL_ERROR",
                                      "Internal server error")
            await response(scope, receive, send_with_context)
        finally:
            # Log the route template instead of request text, URL query, or body.
            route = getattr(scope.get("route"), "path", "<unmatched>")
            logger.info("request_id=%s method=%s route=%s status=%d elapsed_ms=%.3f",
                        request_id, scope["method"], route, status,
                        (perf_counter()-started)*1000)


def install_error_handling(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(HTTPException, http_handler)
    app.add_middleware(RequestContextMiddleware)
