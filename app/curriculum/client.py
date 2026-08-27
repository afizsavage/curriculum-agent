"""Read-only HTTP client for the MBSSE Curriculum Structure API."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from app.config import Settings, get_settings
from app.curriculum.errors import (
    CurriculumAuthError,
    CurriculumInvalidQueryError,
    CurriculumNotFoundError,
    CurriculumTimeoutError,
    CurriculumUnexpectedResponseError,
    CurriculumUnavailableError,
)
from app.logging_utils import get_logger

logger = get_logger(__name__)


class CurriculumAPIClient:
    """Encapsulates base URL, timeouts, retries, validation, and error mapping."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        base = self.settings.resolved_curriculum_api_url() + "/"
        self.base_url = base
        timeout = self.settings.curriculum_api_timeout_seconds
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            transport=transport,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "CurriculumAPIClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> Any:
        cleaned = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {}
        if request_id:
            headers["X-Request-ID"] = request_id
        url_path = path.lstrip("/")
        try:
            response = self._client.get(url_path, params=cleaned, headers=headers)
        except httpx.TimeoutException as exc:
            logger.warning(
                "curriculum_api.timeout path=%r params=%r", url_path, cleaned
            )
            raise CurriculumTimeoutError(
                f"Curriculum API timed out for GET /{url_path}"
            ) from exc
        except httpx.RequestError as exc:
            logger.warning(
                "curriculum_api.unavailable path=%r error=%r", url_path, str(exc)
            )
            raise CurriculumUnavailableError(
                f"Curriculum API unavailable for GET /{url_path}"
            ) from exc

        logger.info(
            "curriculum_api.response path=%r status=%s params=%r",
            url_path,
            response.status_code,
            cleaned,
        )
        return self._parse(response, path=url_path)

    def _parse(self, response: httpx.Response, *, path: str) -> Any:
        status = response.status_code
        if status == 404:
            detail = self._detail(response) or f"Not found: /{path}"
            raise CurriculumNotFoundError(detail)
        if status in {401, 403}:
            raise CurriculumAuthError(
                self._detail(response) or "Curriculum API authentication failed"
            )
        if status == 422:
            raise CurriculumInvalidQueryError(
                self._detail(response) or "Invalid curriculum query"
            )
        if status >= 500:
            raise CurriculumUnavailableError(
                self._detail(response) or f"Curriculum API error ({status})"
            )
        if status >= 400:
            raise CurriculumInvalidQueryError(
                self._detail(response) or f"Curriculum API rejected request ({status})"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise CurriculumUnexpectedResponseError(
                f"Curriculum API returned non-JSON for /{path}"
            ) from exc

    @staticmethod
    def _detail(response: httpx.Response) -> str | None:
        try:
            body = response.json()
        except ValueError:
            return None
        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, str):
                return detail
        return None

    # --- Typed helpers against known Curriculum API routes ---

    def list_curricula(self, **params: Any) -> dict[str, Any]:
        return self.get("api/v1/curricula", params=params)

    def get_curriculum_structure(self, curriculum_id: str) -> dict[str, Any]:
        return self.get(f"api/v1/curricula/{curriculum_id}/structure")

    def list_subjects(self, curriculum_id: str, **params: Any) -> dict[str, Any]:
        return self.get(f"api/v1/curricula/{curriculum_id}/subjects", params=params)

    def get_subject(self, subject_id: str) -> dict[str, Any]:
        return self.get(f"api/v1/subjects/{subject_id}")

    def get_subject_curriculum(self, subject_id: str) -> dict[str, Any]:
        return self.get(f"api/v1/subjects/{subject_id}/curriculum")

    def list_syllabuses(self, **params: Any) -> dict[str, Any]:
        return self.get("api/v1/syllabuses", params=params)

    def list_curriculum_grade_curricula(
        self, curriculum_id: str, **params: Any
    ) -> dict[str, Any]:
        return self.get(
            f"api/v1/curricula/{curriculum_id}/grade-curricula", params=params
        )

    def get_grade_curriculum(self, grade_curriculum_id: str) -> dict[str, Any]:
        return self.get(f"api/v1/grade-curricula/{grade_curriculum_id}")

    def get_grade_curriculum_content(self, grade_curriculum_id: str) -> list[Any]:
        data = self.get(f"api/v1/grade-curricula/{grade_curriculum_id}/content")
        if not isinstance(data, list):
            raise CurriculumUnexpectedResponseError(
                "Expected grade-curriculum content list"
            )
        return data

    def get_curriculum_content(self, content_id: str) -> dict[str, Any]:
        return self.get(f"api/v1/curriculum-content/{content_id}")

    def get_curriculum_content_learning_outcomes(
        self, content_id: str, **params: Any
    ) -> dict[str, Any]:
        return self.get(
            f"api/v1/curriculum-content/{content_id}/learning-outcomes",
            params=params,
        )

    def get_syllabus_content_tree(
        self, syllabus_id: str, *, grade_code: str | None = None
    ) -> list[Any]:
        data = self.get(
            f"api/v1/syllabuses/{syllabus_id}/content/tree",
            params={"grade_code": grade_code},
        )
        if not isinstance(data, list):
            raise CurriculumUnexpectedResponseError(
                "Expected syllabus content tree list"
            )
        return data

    def get_syllabus_learning_outcomes(
        self, syllabus_id: str, *, grade_code: str | None = None
    ) -> list[Any]:
        data = self.get(
            f"api/v1/syllabuses/{syllabus_id}/learning-outcomes",
            params={"grade_code": grade_code},
        )
        if not isinstance(data, list):
            raise CurriculumUnexpectedResponseError(
                "Expected syllabus learning outcomes list"
            )
        return data

    def get_topic(self, topic_id: str) -> dict[str, Any]:
        return self.get(f"api/v1/topics/{topic_id}")

    def get_topic_learning_outcomes(
        self, topic_id: str, *, direct_only: bool = False
    ) -> dict[str, Any]:
        return self.get(
            f"api/v1/topics/{topic_id}/learning-outcomes",
            params={"direct_only": str(direct_only).lower()},
        )

    def get_curriculum_context(self, **params: Any) -> dict[str, Any]:
        return self.get("api/v1/curriculum-context", params=params)

    def resolve_curriculum_context(self, **params: Any) -> dict[str, Any]:
        """V2.1 read-only GradeCurriculum context resolve (additive; not V1)."""
        return self.get("api/v2/curriculum/context/resolve", params=params)

    def resolve_curriculum_id(
        self, *, code: str, version: str | None = None
    ) -> Optional[str]:
        page = self.list_curricula(code=code, limit=20)
        items = page.get("items") if isinstance(page, dict) else None
        if not items:
            return None
        if version:
            for item in items:
                if item.get("version") == version:
                    return str(item["id"])
        return str(items[0]["id"])
