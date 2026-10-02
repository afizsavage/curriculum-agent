"""V2.13G.4 — grade + classification filtering regressions.

Classification is enforced in structured GradeSubject retrieval, not by the LLM.
Authoritative fixture mirrors live MBSSE Primary 3 / Primary 4 catalogue rows.
"""

from __future__ import annotations

import httpx
import pytest

from app.agent.graph_nodes import GraphNodes, filters_from_state
from app.agent.state import CurriculumQAState
from app.agent.verification_checks import run_deterministic_checks
from app.config import Settings
from app.curriculum.client import CurriculumAPIClient
from app.curriculum.codes import extract_filters_from_question
from app.llm.base import LLMMessage
from app.llm.tool_selection import select_tool_calls
from app.schemas.answer import AnswerConfidence, AnswerEvidenceRef
from app.schemas.verification import VerificationRecommendation
from app.tools.curriculum import GetCurriculumStructureTool, GetSubjectTool


CURRICULUM_ID = "11111111-1111-1111-1111-111111111111"
GRADE_ID_C3 = "18fd9dad-5053-4f87-b9e7-220c8adc7d42"
GRADE_ID_C4 = "211b3ef6-fee3-4007-8f9a-c5a1f1118c99"

# Authoritative Primary 3 GradeSubject classifications (live Structure API).
CLASS_3_ASSIGNMENTS: list[tuple[str, str, str]] = [
    ("ENGLISH", "English", "CORE"),
    ("MATHEMATICS", "Mathematics", "CORE"),
    ("CIVIC_EDUCATION", "Civic Education", "CORE"),
    ("SOCIAL_STUDIES", "Social Studies", "CORE"),
    ("ENVIRONMENTAL_LIFE_SKILLS", "Environmental Science", "CORE"),
    ("PRACTICAL_HEALTH_PHYSICAL_EDUCATION", "Physical Health Education", "CORE"),
    ("MUSIC_EXPRESSIVE_ARTS", "Music and Other Expressive Arts", "CORE"),
    ("FRENCH_APPRECIATION", "French Appreciation", "CORE"),
    ("HOME_ECONOMICS", "Home Economics", "AVAILABLE"),
    ("PRACTICAL_ARTS", "Practical Arts", "AVAILABLE"),
    ("RELIGIOUS_MORAL_EDUCATION", "Religious and Moral Education", "AVAILABLE"),
    ("SIERRA_LEONE_NATIONAL_LANGUAGES", "Sierra Leone National Languages", "AVAILABLE"),
    ("ICT_LITERACY", "ICT Literacy", "AVAILABLE"),
    ("AGRICULTURAL_SCIENCE", "Agricultural Science", "AVAILABLE"),
]

# Primary 4 — ICT_LITERACY is CORE here (AVAILABLE in Primary 3) for cross-grade Test G.
CLASS_4_ASSIGNMENTS: list[tuple[str, str, str]] = [
    ("ENGLISH_LANGUAGE", "English Language", "CORE"),
    ("MATHEMATICS", "Mathematics", "CORE"),
    ("CIVIC_EDUCATION", "Civic Education", "CORE"),
    ("SOCIAL_STUDIES", "Social Studies", "CORE"),
    ("PRACTICAL_HEALTH_PHYSICAL_EDUCATION", "Physical Health Education", "CORE"),
    ("MUSIC_EXPRESSIVE_ARTS", "Music and Other Expressive Arts", "CORE"),
    ("FRENCH_APPRECIATION", "French Appreciation", "CORE"),
    ("SCIENCE", "Science", "CORE"),
    ("ICT_LITERACY", "ICT Literacy", "CORE"),
    ("HOME_ECONOMICS", "Home Economics", "AVAILABLE"),
    ("PRACTICAL_ARTS", "Practical Arts", "AVAILABLE"),
    ("RELIGIOUS_MORAL_EDUCATION", "Religious and Moral Education", "AVAILABLE"),
    ("SIERRA_LEONE_NATIONAL_LANGUAGES", "Sierra Leone National Languages", "AVAILABLE"),
    ("AGRICULTURAL_SCIENCE", "Agricultural Science", "AVAILABLE"),
]

CLASS_3_CORE_CODES = {c for c, _, cls in CLASS_3_ASSIGNMENTS if cls == "CORE"}
CLASS_3_NON_CORE_CODES = {c for c, _, cls in CLASS_3_ASSIGNMENTS if cls != "CORE"}
CLASS_3_ALL_CODES = {c for c, _, _ in CLASS_3_ASSIGNMENTS}
CLASS_4_CORE_CODES = {c for c, _, cls in CLASS_4_ASSIGNMENTS if cls == "CORE"}


def _assignment_items(
    grade_id: str, assignments: list[tuple[str, str, str]]
) -> list[dict]:
    items = []
    for order, (code, name, classification) in enumerate(assignments, start=1):
        sid = f"sub-{code.lower()}"
        items.append(
            {
                "id": f"gs-{grade_id[:8]}-{code}",
                "curriculum_id": CURRICULUM_ID,
                "grade_id": grade_id,
                "subject_id": sid,
                "display_order": order,
                "weekly_periods": 5,
                "status": "ACTIVE",
                "classification": classification,
                "subject": {"id": sid, "code": code, "name": name},
            }
        )
    return items


ALL_GRADE_SUBJECTS = _assignment_items(GRADE_ID_C3, CLASS_3_ASSIGNMENTS) + _assignment_items(
    GRADE_ID_C4, CLASS_4_ASSIGNMENTS
)


def _classification_router(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/api/v1/curricula"):
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": CURRICULUM_ID,
                        "code": "MBSSE-BEC",
                        "version": "2020",
                        "name": "BEC",
                    }
                ],
                "total": 1,
                "limit": 20,
                "offset": 0,
            },
        )
    if path.endswith(f"/api/v1/curricula/{CURRICULUM_ID}/structure"):
        return httpx.Response(
            200,
            json={
                "id": CURRICULUM_ID,
                "code": "MBSSE-BEC",
                "education_levels": [
                    {
                        "id": "lvl",
                        "name": "Primary",
                        "grades": [
                            {
                                "id": GRADE_ID_C3,
                                "code": "CLASS_3",
                                "name": "Class 3",
                                "subjects": [
                                    {
                                        "id": f"sub-{c.lower()}",
                                        "code": c,
                                        "name": n,
                                    }
                                    for c, n, _ in CLASS_3_ASSIGNMENTS
                                ],
                            },
                            {
                                "id": GRADE_ID_C4,
                                "code": "CLASS_4",
                                "name": "Class 4",
                                "subjects": [
                                    {
                                        "id": f"sub-{c.lower()}",
                                        "code": c,
                                        "name": n,
                                    }
                                    for c, n, _ in CLASS_4_ASSIGNMENTS
                                ],
                            },
                        ],
                    }
                ],
            },
        )
    if path.endswith(f"/api/v1/curricula/{CURRICULUM_ID}/grade-subjects"):
        items = list(ALL_GRADE_SUBJECTS)
        grade_id = request.url.params.get("grade_id")
        classification = request.url.params.get("classification")
        subject_id = request.url.params.get("subject_id")
        if grade_id:
            items = [i for i in items if i["grade_id"] == grade_id]
        if classification:
            items = [i for i in items if i["classification"] == classification]
        if subject_id:
            items = [i for i in items if i["subject_id"] == subject_id]
        return httpx.Response(
            200,
            json={"items": items, "total": len(items), "limit": 200, "offset": 0},
        )
    return httpx.Response(404, json={"detail": f"unhandled {path}"})


@pytest.fixture
def client() -> CurriculumAPIClient:
    settings = Settings(curriculum_api_base_url="http://curriculum.test")
    return CurriculumAPIClient(
        settings=settings, transport=httpx.MockTransport(_classification_router)
    )


def _codes(subjects: list[dict]) -> set[str]:
    return {str(s["code"]).upper() for s in subjects}


# --- Filter extraction / understand ---


def test_extract_primary3_core_filters():
    filters = extract_filters_from_question(
        "What are the core subjects in Primary 3?"
    )
    assert filters["grade"] == "CLASS_3"
    assert filters["classification"] == "CORE"
    assert filters["subject"] is None


def test_extract_primary3_all_subjects_no_classification():
    filters = extract_filters_from_question(
        "What subjects are taught in Primary 3?"
    )
    assert filters["grade"] == "CLASS_3"
    assert filters["classification"] is None


def test_extract_non_core_and_missing_grade():
    filters = extract_filters_from_question(
        "What are the non-core subjects in Primary 3?"
    )
    assert filters["grade"] == "CLASS_3"
    assert filters["classification"] == "NON_CORE"

    missing = extract_filters_from_question("What are the core subjects?")
    assert missing["grade"] is None
    assert missing["classification"] == "CORE"


def test_understand_sets_classification_on_state():
    from unittest.mock import MagicMock

    nodes = GraphNodes(
        settings=Settings(),
        retrieval=MagicMock(),
        answer_node=MagicMock(),
        verification_node=MagicMock(),
    )
    qa = CurriculumQAState.initial(
        question="What are the core subjects in Primary 3?"
    )
    result = nodes.understand({"qa": qa, "prior_filters": {}, "visited_nodes": []})
    assert result["qa"].grade == "CLASS_3"
    assert result["qa"].classification == "CORE"
    assert filters_from_state(result["qa"])["classification"] == "CORE"


# --- Structured retrieval (deterministic layer) ---


def test_a_primary3_core_structured_retrieval(client):
    tool = GetCurriculumStructureTool(client)
    result = tool.execute(grade="Primary 3", classification="CORE")
    assert result.success
    codes = _codes(result.data["subjects"])
    assert codes == CLASS_3_CORE_CODES
    assert CLASS_3_NON_CORE_CODES.isdisjoint(codes)
    assert result.data["classification"] == "CORE"
    assert all(s.get("classification") == "CORE" for s in result.data["subjects"])


def test_b_primary3_all_subjects_structured_retrieval(client):
    tool = GetCurriculumStructureTool(client)
    result = tool.execute(grade="Primary 3")
    assert result.success
    codes = _codes(result.data["subjects"])
    assert codes == CLASS_3_ALL_CODES
    assert result.data.get("classification") is None


def test_c_primary3_non_core_structured_retrieval(client):
    tool = GetCurriculumStructureTool(client)
    result = tool.execute(grade="Primary 3", classification="NON_CORE")
    assert result.success
    codes = _codes(result.data["subjects"])
    assert codes == CLASS_3_NON_CORE_CODES
    assert CLASS_3_CORE_CODES.isdisjoint(codes)
    assert all(s.get("classification") != "CORE" for s in result.data["subjects"])


def test_d_mathematics_classification_within_primary3(client):
    tool = GetSubjectTool(client)
    result = tool.execute(grade="Primary 3", subject="Mathematics")
    assert result.success
    assert result.data["grade"] == "CLASS_3"
    assert result.data["classification"] == "CORE"
    assert result.data["subject"]["code"] == "MATHEMATICS"
    assert result.data["subject"]["classification"] == "CORE"


def test_e_primary4_core_isolated_from_primary3(client):
    tool = GetCurriculumStructureTool(client)
    result = tool.execute(grade="Primary 4", classification="CORE")
    assert result.success
    codes = _codes(result.data["subjects"])
    assert codes == CLASS_4_CORE_CODES
    # Primary 3-only CORE code must not appear
    assert "ENGLISH" not in codes  # Primary 4 uses ENGLISH_LANGUAGE
    assert "ENVIRONMENTAL_LIFE_SKILLS" not in codes
    assert all(s.get("classification") == "CORE" for s in result.data["subjects"])


def test_g_cross_grade_classification_for_same_subject(client):
    """ICT_LITERACY classification differs by grade — must not use a global label."""
    tool = GetSubjectTool(client)
    c3 = tool.execute(grade="Primary 3", subject="ICT Literacy")
    c4 = tool.execute(grade="Primary 4", subject="ICT Literacy")
    assert c3.success and c4.success
    assert c3.data["classification"] == "AVAILABLE"
    assert c4.data["classification"] == "CORE"
    assert c3.data["grade"] == "CLASS_3"
    assert c4.data["grade"] == "CLASS_4"


# --- Natural-language tool selection / clarify ---


def test_tool_selection_passes_classification_for_core_subjects():
    tools = [
        {
            "name": "get_curriculum_structure",
            "description": "",
            "parameters": {},
        }
    ]
    calls = select_tool_calls(
        [LLMMessage(role="user", content="Question: What are the core subjects in Primary 3?")],
        tools,
    )
    assert len(calls) == 1
    assert calls[0].name == "get_curriculum_structure"
    assert calls[0].arguments["grade"] == "CLASS_3"
    assert calls[0].arguments["classification"] == "CORE"


def test_f_missing_grade_does_not_select_subjects_tool():
    tools = [
        {
            "name": "get_curriculum_structure",
            "description": "",
            "parameters": {},
        }
    ]
    calls = select_tool_calls(
        [LLMMessage(role="user", content="Question: What are the core subjects?")],
        tools,
    )
    assert calls == []

    state = CurriculumQAState.initial(question="What are the core subjects?")
    state.classification = "CORE"
    state.draft_answer = "Here are some core subjects: Mathematics, English."
    state.final_answer = state.draft_answer
    state.answer_confidence = AnswerConfidence.MEDIUM
    state.answer_evidence = [
        AnswerEvidenceRef(
            claim="Mathematics is a core subject",
            entity_type="subject",
            entity_id="x",
            name="Mathematics",
        )
    ]
    result = run_deterministic_checks(state)
    assert result.recommendation == VerificationRecommendation.CLARIFY
    assert result.clarification
    assert "grade" in result.clarification.lower() or "level" in result.clarification.lower()


def test_tool_selection_classification_of_mathematics():
    tools = [
        {"name": "get_subject", "description": "", "parameters": {}},
        {"name": "get_curriculum_structure", "description": "", "parameters": {}},
    ]
    calls = select_tool_calls(
        [
            LLMMessage(
                role="user",
                content="Question: What is the classification of Mathematics in Primary 3?",
            )
        ],
        tools,
    )
    assert len(calls) == 1
    assert calls[0].name == "get_subject"
    assert calls[0].arguments["grade"] == "CLASS_3"
    assert calls[0].arguments["subject"] == "MATHEMATICS"
