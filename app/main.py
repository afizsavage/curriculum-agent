from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.v1 import api_router
from app.exceptions import AgentError
from app.logging_utils import configure_logging, get_logger, new_request_id

logger = get_logger(__name__)


def create_app() -> FastAPI:
    configure_logging()
    application = FastAPI(
        title="Curriculum Q&A Agent",
        description=(
            "Independent MBSSE Curriculum Q&A Agent service. "
            "Phase 2 retrieves authoritative curriculum evidence from the Curriculum "
            "Structure API via read-only tools. Answer generation and verification "
            "arrive in Phase 3. This service does not store or mutate curriculum data."
        ),
        version=__version__,
    )

    @application.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or new_request_id()
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @application.exception_handler(AgentError)
    async def handle_agent_error(request: Request, exc: AgentError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "agent.error code=%r detail=%r request_id=%r",
            exc.code,
            exc.message,
            request_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.message,
                "code": exc.code,
                "request_id": request_id,
            },
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Invalid request",
                "code": "INVALID_REQUEST",
                "request_id": request_id,
                "errors": jsonable_encoder(exc.errors()),
            },
        )

    @application.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unexpected.error request_id=%r", request_id)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Unexpected agent error",
                "code": "UNEXPECTED_ERROR",
                "request_id": request_id,
            },
        )

    @application.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "curriculum-agent", "version": __version__}

    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix="/api/v1")
    return application


app = create_app()
