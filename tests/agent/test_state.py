from app.agent.state import CurriculumQAState, PlanStep, RetrievedContextItem
from app.enums import AgentStatus


def test_state_initializes_with_defaults():
    state = CurriculumQAState.initial(question="What is taught in P4 Math?")
    assert state.question == "What is taught in P4 Math?"
    assert state.conversation_id
    assert state.intent is None
    assert state.level is None
    assert state.grade is None
    assert state.subject is None
    assert state.topic is None
    assert state.plan is None
    assert state.retrieved_context == []
    assert state.draft_answer is None
    assert state.verification is None
    assert state.iteration == 0
    assert state.tool_calls == 0
    assert state.status == AgentStatus.RECEIVED
    assert state.error is None


def test_state_preserves_conversation_id():
    state = CurriculumQAState.initial(
        question="Follow-up",
        conversation_id="11111111-1111-1111-1111-111111111111",
    )
    assert state.conversation_id == "11111111-1111-1111-1111-111111111111"


def test_state_can_be_updated():
    state = CurriculumQAState.initial(question="Q")
    state.intent = "list_topics"
    state.grade = "CLASS_4"
    state.subject = "MATHEMATICS"
    state.plan = [PlanStep(id="1", description="retrieve topics")]
    state.retrieved_context.append(
        RetrievedContextItem(source="stub", content="Fractions")
    )
    state.draft_answer = "Fractions are taught."
    state.bump_iteration()
    state.bump_tool_calls(2)
    state.status = AgentStatus.ANSWERING
    assert state.iteration == 1
    assert state.tool_calls == 2
    assert state.grade == "CLASS_4"
    assert len(state.retrieved_context) == 1


def test_mark_failed():
    state = CurriculumQAState.initial(question="Q")
    state.mark_failed("boom")
    assert state.status == AgentStatus.FAILED
    assert state.error == "boom"
