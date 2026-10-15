import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent
from schemas import AnswerResponse, CalculationResponse, SummarizationResponse, UserIntent
from conftest import FakeLLM


def make_config(llm, thread_id="test-thread"):
    return {"configurable": {"thread_id": thread_id, "llm": llm, "tools": []}}


def run_turn(workflow, llm, user_input="hello", thread_id="test-thread"):
    return workflow.invoke(
        {"user_input": user_input, "messages": [], "actions_taken": []},
        make_config(llm, thread_id),
    )


@pytest.mark.parametrize("intent_type, next_step", [
    ("qa", "qa_agent"),
    ("summarization", "summarization_agent"),
    ("calculation", "calculation_agent"),
    ("unknown", "qa_agent"),
])
def test_classify_intent_sets_next_step(intent_type, next_step):
    llm = FakeLLM(intent_type=intent_type)

    update = agent.classify_intent({"user_input": "hi", "messages": []}, make_config(llm))

    assert update["next_step"] == next_step
    assert update["intent"].intent_type == intent_type
    assert update["actions_taken"] == ["classify_intent"]


def test_classify_intent_includes_history_in_prompt():
    llm = FakeLLM()
    history = [HumanMessage(content="Tell me about INV-001"), AIMessage(content="It totals $22,000")]

    agent.classify_intent({"user_input": "And INV-002?", "messages": history}, make_config(llm))

    schema, prompt = llm.prompts[0]
    assert schema is UserIntent
    assert "And INV-002?" in prompt
    assert "Tell me about INV-001" in prompt
    assert "It totals $22,000" in prompt


def test_classify_intent_handles_empty_history():
    llm = FakeLLM()

    agent.classify_intent({"user_input": "hi", "messages": []}, make_config(llm))

    assert "No previous conversation." in llm.prompts[0][1]


def test_update_memory_returns_summary_and_documents():
    llm = FakeLLM(summary="Discussed INV-001", document_ids=["INV-001"])
    state = {
        "messages": [HumanMessage(content="What is INV-001?"), AIMessage(content="An invoice")],
        "conversation_summary": "Earlier summary",
    }

    update = agent.update_memory(state, make_config(llm))

    assert update == {
        "conversation_summary": "Discussed INV-001",
        "active_documents": ["INV-001"],
        "actions_taken": ["update_memory"],
        "next_step": "end",
    }
    prompt_text = llm.prompts[0][1].to_string()
    assert "Earlier summary" in prompt_text
    assert "User: What is INV-001?" in prompt_text
    assert "Assistant: An invoice" in prompt_text


def test_update_memory_skips_tool_messages_in_transcript():
    llm = FakeLLM()
    state = {
        "messages": [
            HumanMessage(content="Read INV-001"),
            ToolMessage(content="RAW TOOL OUTPUT", name="document_reader", tool_call_id="1"),
            AIMessage(content="Done"),
        ],
        "conversation_summary": "",
    }

    agent.update_memory(state, make_config(llm))

    assert "RAW TOOL OUTPUT" not in llm.prompts[0][1].to_string()


def test_invoke_react_agent_counts_only_new_tool_calls(monkeypatch):
    history = [ToolMessage(content="old", name="document_reader", tool_call_id="old-1")]
    new = [ToolMessage(content="new", name="calculator", tool_call_id="new-1"), AIMessage(content="done")]

    class FakeAgent:
        def invoke(self, payload):
            return {"messages": payload["messages"] + new}

    class FakeChatModel:
        def bind_tools(self, tools):
            return self

    monkeypatch.setattr(agent, "create_react_agent", lambda **kwargs: FakeAgent())

    _, tools_used = agent.invoke_react_agent(AnswerResponse, history, FakeChatModel(), [])

    assert tools_used == ["calculator"]


@pytest.mark.parametrize("intent_type, node, schema", [
    ("qa", "qa_agent", AnswerResponse),
    ("summarization", "summarization_agent", SummarizationResponse),
    ("calculation", "calculation_agent", CalculationResponse),
    ("unknown", "qa_agent", AnswerResponse),
])
def test_workflow_routes_each_intent(fake_react_agent, intent_type, node, schema):
    workflow = agent.create_workflow(None, [])

    final_state = run_turn(workflow, FakeLLM(intent_type=intent_type))

    assert final_state["actions_taken"] == ["classify_intent", node, "update_memory"]
    assert fake_react_agent == [schema]
    assert final_state["messages"][-1].content == f"{schema.__name__} answer"
    assert final_state["tools_used"] == ["document_reader"]
    assert final_state["active_documents"] == ["INV-001"]
    assert final_state["next_step"] == "end"


def test_workflow_persists_state_across_turns(fake_react_agent):
    workflow = agent.create_workflow(None, [])
    llm = FakeLLM(summary="First turn summary")

    run_turn(workflow, llm, user_input="first question")
    saved = workflow.get_state(make_config(llm)).values
    assert saved["conversation_summary"] == "First turn summary"

    run_turn(workflow, llm, user_input="second question")
    contents = [m.content for m in workflow.get_state(make_config(llm)).values["messages"]]
    assert "first question" in contents
    assert "second question" in contents


def test_workflow_keeps_threads_separate(fake_react_agent):
    workflow = agent.create_workflow(None, [])
    llm = FakeLLM()

    run_turn(workflow, llm, user_input="only in thread A", thread_id="A")
    run_turn(workflow, llm, user_input="only in thread B", thread_id="B")

    contents_b = [m.content for m in workflow.get_state(make_config(llm, "B")).values["messages"]]
    assert "only in thread B" in contents_b
    assert "only in thread A" not in contents_b


def test_checkpointer_restores_schema_objects_without_warnings(fake_react_agent, recwarn, caplog):
    workflow = agent.create_workflow(None, [])
    llm = FakeLLM(intent_type="calculation")

    run_turn(workflow, llm)
    run_turn(workflow, llm)
    intent = workflow.get_state(make_config(llm)).values["intent"]

    assert isinstance(intent, UserIntent)
    assert "unregistered type" not in caplog.text
    assert not [w for w in recwarn if "unregistered type" in str(w.message)]
