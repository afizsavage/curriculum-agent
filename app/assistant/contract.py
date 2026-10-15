"""Internal contracts for the early-release teacher assistant.

These types sit beside the public curriculum QA API. They embed ``AskResponse``
when a QA turn produced one. They do not replace or extend that schema.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.agent import AskResponse


class AssistantRoute(str, Enum):
    """Closed destinations. Code owns dispatch; classifiers may only name these."""

    CURRICULUM = "curriculum"
    GENERAL = "general"
    MIXED = "mixed"
    CLARIFY = "clarify"


CLOSED_ROUTES = frozenset(route.value for route in AssistantRoute)

CURRICULUM_SOURCE = "mbsse_curriculum"
GENERAL_SOURCE = "general_ai_assistance"
GENERAL_LABEL = "General AI assistance"

CoverageDecision = Literal["accept", "reject", "unavailable"]
RouteSource = Literal["heuristic", "classifier"]
FailureKind = Literal["timeout", "execution_failure", "provider_error"]


class InvalidAssistantRoute(ValueError):
    """Classifier output was outside the closed route set.

    Callers must not substitute ``general`` or invent a destination.
    """

    def __init__(self, raw: object) -> None:
        self.raw = raw
        super().__init__(
            "Assistant route must be one of: curriculum, general, mixed, clarify. "
            f"Received {raw!r}."
        )


class AssistantTurnRequest(BaseModel):
    """One teacher turn.

    This request does not accept a teacher id. Ownership lives on the thread
    store and requires a principal issued after real authentication.
    """

    question: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    request_id: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    task: Optional[str] = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class RoutingResult(BaseModel):
    route: AssistantRoute
    source: RouteSource
    reason: str
    raw_classification: Optional[str] = None


class CurriculumExecutionFailure(BaseModel):
    """QA raised before an ``AskResponse`` existed."""

    kind: FailureKind
    code: str
    message: str
    http_status: int


class GeneralExecutionFailure(BaseModel):
    """General-answer generation failed. This is not a curriculum result."""

    code: str
    message: str
    http_status: int


class CurriculumOutcome(BaseModel):
    """Fake or future in-process QA result consumed by the composer.

    ``coverage_decision`` is read from in-process verifier metadata when a
    caller has the QA state. ``AskResponse.verification`` does not include it.
    """

    response: Optional[AskResponse] = None
    failure: Optional[CurriculumExecutionFailure] = None
    coverage_decision: Optional[CoverageDecision] = None
    http_status: Optional[int] = None


class CurriculumChannel(BaseModel):
    """Official MBSSE curriculum section. Text is copied from QA."""

    source: Literal["mbsse_curriculum"] = CURRICULUM_SOURCE
    response: Optional[AskResponse] = None
    failure: Optional[CurriculumExecutionFailure] = None
    coverage_decision: Optional[CoverageDecision] = None
    http_status: Optional[int] = None
    technical_failure: bool = False


class GeneralChannel(BaseModel):
    """Non-curriculum teaching help. Never an official curriculum answer."""

    source: Literal["general_ai_assistance"] = GENERAL_SOURCE
    label: str = GENERAL_LABEL
    text: str


class AssistantTurnResponse(BaseModel):
    route: AssistantRoute
    routing: RoutingResult
    curriculum: Optional[CurriculumChannel] = None
    general: Optional[GeneralChannel] = None
    general_failure: Optional[GeneralExecutionFailure] = None
    clarification: Optional[str] = None
    general_suppressed_reason: Optional[str] = None

    @property
    def official_curriculum_answer(self) -> Optional[str]:
        if self.curriculum is None or self.curriculum.response is None:
            return None
        return self.curriculum.response.answer
