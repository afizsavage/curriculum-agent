from app.agent.context import ConversationStore
from app.agent.state import CurriculumQAState
from app.enums import MessageRole


def test_conversation_store_creates_and_reuses():
    store = ConversationStore()
    first = store.get_or_create(None)
    second = store.get_or_create(first.conversation_id)
    assert first.conversation_id == second.conversation_id
    assert store.get(first.conversation_id) is second


def test_conversation_messages_and_state():
    store = ConversationStore()
    ctx = store.get_or_create(None)
    ctx.append_user("What topics are in Primary 4 Mathematics?")
    state = CurriculumQAState.initial(
        question=ctx.current_question or "",
        conversation_id=ctx.conversation_id,
    )
    ctx.set_state(state)
    store.save(ctx)
    loaded = store.get(ctx.conversation_id)
    assert loaded is not None
    assert loaded.messages[0].role == MessageRole.USER
    assert loaded.current_state is not None
    assert loaded.current_state.question.startswith("What topics")
