from app.llm.base import LLMMessage
from app.llm.tool_selection import select_tool_calls


TOOLS = [
    {"name": "search_curriculum", "description": "", "parameters": {}},
    {"name": "get_curriculum_structure", "description": "", "parameters": {}},
    {"name": "get_subject", "description": "", "parameters": {}},
    {"name": "get_topic", "description": "", "parameters": {}},
    {"name": "get_learning_objectives", "description": "", "parameters": {}},
]


def _calls(question: str):
    messages = [
        LLMMessage(
            role="user",
            content=f"Question: {question}\nKnown filters: {{}}",
        )
    ]
    return select_tool_calls(messages, TOOLS)


def test_select_structure_for_topics():
    calls = _calls("What topics are in Primary 4 Mathematics?")
    assert calls[0].name == "get_curriculum_structure"
    assert calls[0].arguments["grade"] == "CLASS_4"
    assert calls[0].arguments["subject"] == "MATHEMATICS"


def test_select_structure_for_subjects_available():
    calls = _calls("What subjects are available in Primary 4?")
    assert calls[0].name == "get_curriculum_structure"
    assert "subject" not in calls[0].arguments or not calls[0].arguments.get("subject")


def test_select_topic():
    calls = _calls("What is the fractions topic in Primary 4 Mathematics?")
    assert calls[0].name == "get_topic"


def test_select_learning_objectives():
    calls = _calls(
        "What are the learning objectives for fractions in Primary 4 Mathematics?"
    )
    assert calls[0].name == "get_learning_objectives"


def test_select_search_for_measurement():
    calls = _calls("Find curriculum content related to measurement in Primary 4 Mathematics.")
    assert calls[0].name == "search_curriculum"
    assert "measurement" in calls[0].arguments["query"].lower()
