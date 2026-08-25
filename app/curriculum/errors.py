"""Curriculum API and agent error extensions for Phase 2."""

from app.exceptions import AgentError


class CurriculumAPIError(AgentError):
    """Base error for Curriculum Structure API client failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 502,
        code: str = "CURRICULUM_API_ERROR",
    ) -> None:
        super().__init__(message, status_code=status_code, code=code)


class CurriculumNotFoundError(CurriculumAPIError):
    def __init__(self, message: str = "Curriculum entity not found") -> None:
        super().__init__(message, status_code=404, code="CURRICULUM_NOT_FOUND")


class CurriculumUnavailableError(CurriculumAPIError):
    def __init__(self, message: str = "Curriculum API unavailable") -> None:
        super().__init__(message, status_code=503, code="CURRICULUM_UNAVAILABLE")


class CurriculumTimeoutError(CurriculumAPIError):
    def __init__(self, message: str = "Curriculum API request timed out") -> None:
        super().__init__(message, status_code=504, code="CURRICULUM_TIMEOUT")


class CurriculumAuthError(CurriculumAPIError):
    def __init__(self, message: str = "Curriculum API authentication failed") -> None:
        super().__init__(message, status_code=401, code="CURRICULUM_AUTH_FAILURE")


class CurriculumInvalidQueryError(CurriculumAPIError):
    def __init__(self, message: str = "Invalid curriculum query") -> None:
        super().__init__(message, status_code=422, code="CURRICULUM_INVALID_QUERY")


class CurriculumUnexpectedResponseError(CurriculumAPIError):
    def __init__(self, message: str = "Unexpected Curriculum API response") -> None:
        super().__init__(
            message, status_code=502, code="CURRICULUM_UNEXPECTED_RESPONSE"
        )
