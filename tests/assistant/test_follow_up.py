"""Context follow-ups use the owned thread and do not invent curriculum facts."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.assistant.chat import configure_assistant_runtime, execute_chat, reset_assistant_runtime
from app.assistant.identity import (
    VerifiedTeacherClaims,
    bind_verified_claims,
    current_identity_gate,
    register_identity_gate,
    reset_identity_gate,
)
from app.assistant.threads import AssistantThreadStore, ThreadStoreError
from app.config import Settings
from app.curriculum.evidence import EvidenceStatus
from app.enums import AgentStatus
from app.schemas.assistant import AssistantChatRequest
from app.schemas.verification import VerificationRecommendation, VerificationStatus
from tests.assistant.test_qa_adapter import FakeQAAgent, _state

_CURRICULUM = "What topics are taught in Primary 4 Mathematics?"
_GENERAL = "How should I settle a noisy class before the lesson starts?"
_ADVICE = "Agree a quiet start before the lesson begins."
_GAP = "I couldn't find sufficient MBSSE curriculum evidence to answer this reliably."


class _Gate:
    def __init__(self, teacher_id: str) -> None:
        self.teacher_id = teacher_id

    def verify(self, request: object) -> VerifiedTeacherClaims:
        return VerifiedTeacherClaims(self.teacher_id, None)


class _Calls:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def __call__(self, request) -> str:
        self.questions.append(request.question)
        return _ADVICE


@pytest.fixture(autouse=True)
def _reset_boundaries():
    reset_identity_gate()
    reset_assistant_runtime()
    yield
    reset_identity_gate()
    reset_assistant_runtime()


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="development",
        assistant_enabled=False,
        assistant_local_stub_identity=False,
        assistant_thread_sqlite_path=str(tmp_path / "assistant_threads.sqlite"),
        llm_provider="stub",
    )


def _request() -> object:
    state = type("State", (), {"request_id": "follow-up"})()
    return type("Request", (), {"state": state})()


def _completed_agent() -> FakeQAAgent:
    return FakeQAAgent(
        _state(
            answer="Primary 4 Mathematics includes Number and Numeration.",
            status=AgentStatus.COMPLETED,
            evidence_status=EvidenceStatus.FOUND,
            verification_status=VerificationStatus.PASSED,
            recommendation=VerificationRecommendation.ACCEPT,
            passed=True,
            coverage_decision="accept",
        )
    )


def test_follow_up_after_a_curriculum_answer_keeps_grade_and_subject(tmp_path: Path):
    register_identity_gate(_Gate("teacher-a"))
    agent = _completed_agent()
    general = _Calls()
    configure_assistant_runtime(qa_agent=agent, general=general)
    settings = _settings(tmp_path)
    first = execute_chat(
        AssistantChatRequest(question=_CURRICULUM),
        _request(),
        settings=settings,
    )
    second = execute_chat(
        AssistantChatRequest(question="Tell me more", thread_id=first.thread_id),
        _request(),
        settings=settings,
    )

    assert second.route == "curriculum"
    assert second.general_text is None
    assert second.curriculum is not None
    assert agent.calls[-1]["conversation_id"] is None
    sent = agent.calls[-1]["question"]
    assert sent != _CURRICULUM
    assert "primary 4" in sent
    assert "mathematics" in sent
    assert "Number and Numeration" not in sent
    store = AssistantThreadStore(tmp_path / "assistant_threads.sqlite")
    try:
        record = store.get_thread(
            bind_verified_claims(
                VerifiedTeacherClaims("teacher-a"),
                gate=current_identity_gate(),
            ),
            first.thread_id,
        )
    finally:
        store.close()
    assert [message.role for message in record.messages[:4]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert record.messages[0].content == _CURRICULUM
    assert record.messages[2].content == "Tell me more"
    assert general.questions == []


def test_follow_up_after_general_teaching_does_not_call_qa(tmp_path: Path):
    register_identity_gate(_Gate("teacher-a"))
    agent = _completed_agent()
    general = _Calls()
    configure_assistant_runtime(qa_agent=agent, general=general)
    settings = _settings(tmp_path)
    first = execute_chat(
        AssistantChatRequest(question=_GENERAL),
        _request(),
        settings=settings,
    )
    second = execute_chat(
        AssistantChatRequest(question="Tell me more", thread_id=first.thread_id),
        _request(),
        settings=settings,
    )
    assert second.route == "general"
    assert second.curriculum is None
    assert second.general_text == _ADVICE
    assert agent.calls == []
    assert "settle a noisy" in general.questions[-1]
    assert general.questions[-1] != _GENERAL


def test_follow_up_without_context_asks_for_clarification(tmp_path: Path):
    register_identity_gate(_Gate("teacher-a"))
    agent = _completed_agent()
    general = _Calls()
    configure_assistant_runtime(qa_agent=agent, general=general)
    response = execute_chat(
        AssistantChatRequest(question="Tell me more"),
        _request(),
        settings=_settings(tmp_path),
    )
    assert response.route == "clarify"
    assert response.curriculum is None
    assert response.general_text is None
    assert response.clarification
    assert agent.calls == []
    assert general.questions == []


def test_follow_up_does_not_read_another_teachers_thread(tmp_path: Path):
    gate = _Gate("teacher-a")
    register_identity_gate(gate)
    configure_assistant_runtime(qa_agent=_completed_agent(), general=_Calls())
    settings = _settings(tmp_path)
    first = execute_chat(
        AssistantChatRequest(question=_CURRICULUM),
        _request(),
        settings=settings,
    )
    gate.teacher_id = "teacher-b"
    agent = _completed_agent()
    general = _Calls()
    configure_assistant_runtime(qa_agent=agent, general=general)
    with pytest.raises(Exception) as raised:
        execute_chat(
            AssistantChatRequest(question="Tell me more", thread_id=first.thread_id),
            _request(),
            settings=settings,
        )
    assert type(raised.value).__name__ == "ThreadUnavailable"
    assert agent.calls == []
    assert general.questions == []


def test_follow_up_after_insufficient_evidence_does_not_add_general_advice(tmp_path: Path):
    register_identity_gate(_Gate("teacher-a"))
    gap = FakeQAAgent(
        _state(
            answer=_GAP,
            status=AgentStatus.INSUFFICIENT_EVIDENCE,
            evidence_status=EvidenceStatus.NOT_FOUND,
            verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            recommendation=VerificationRecommendation.FALLBACK,
            passed=False,
        )
    )
    general = _Calls()
    configure_assistant_runtime(qa_agent=gap, general=general)
    settings = _settings(tmp_path)
    first = execute_chat(
        AssistantChatRequest(
            question=(
                "What does the Class 4 Mathematics syllabus say about fractions, "
                "and how should I teach it?"
            )
        ),
        _request(),
        settings=settings,
    )
    assert first.general_text is None
    second = execute_chat(
        AssistantChatRequest(question="Tell me more", thread_id=first.thread_id),
        _request(),
        settings=settings,
    )
    assert second.route == "curriculum"
    assert second.general_text is None
    assert general.questions == []
    sent = gap.calls[-1]["question"]
    assert "class 4" in sent
    assert "mathematics" in sent
    assert "how should i teach" not in sent
    assert "fractions of a whole" not in sent


def test_context_load_and_append_failures_do_not_call_handlers(tmp_path: Path, monkeypatch):
    register_identity_gate(_Gate("teacher-a"))
    agent = _completed_agent()
    general = _Calls()
    configure_assistant_runtime(qa_agent=agent, general=general)
    settings = _settings(tmp_path)
    first = execute_chat(
        AssistantChatRequest(question=_CURRICULUM),
        _request(),
        settings=settings,
    )
    agent.calls.clear()

    def fail_load(self, principal, thread_id):
        raise ThreadStoreError("Assistant thread store is unavailable")

    monkeypatch.setattr(AssistantThreadStore, "get_thread", fail_load)
    with pytest.raises(ThreadStoreError):
        execute_chat(
            AssistantChatRequest(question="Tell me more", thread_id=first.thread_id),
            _request(),
            settings=settings,
        )
    assert agent.calls == []
    assert general.questions == []

    monkeypatch.undo()

    def fail_append(self, principal, thread_id, role, content):
        raise ThreadStoreError("Assistant thread store is unavailable")

    monkeypatch.setattr(AssistantThreadStore, "append_message", fail_append)
    with pytest.raises(ThreadStoreError):
        execute_chat(
            AssistantChatRequest(question="Tell me more", thread_id=first.thread_id),
            _request(),
            settings=settings,
        )
    assert agent.calls == []
    assert general.questions == []
