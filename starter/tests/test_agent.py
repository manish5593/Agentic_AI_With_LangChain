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
        "messages": [
            HumanMessage(content="What is INV-001?"),
            ToolMessage(content="Document INV-001:\n\nInvoice", name="document_reader", tool_call_id="1"),
            AIMessage(content="An invoice"),
        ],
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


def reader_message(doc_id, call_id="r"):
    return ToolMessage(content=f"Document {doc_id}:\n\ncontent", name="document_reader", tool_call_id=call_id)


def test_document_ids_from_tools():
    messages = [
        HumanMessage(content="find invoices"),
        ToolMessage(
            content="Found 2 document(s):\n\nDocument 1 (ID: INV-003):\n...\nDocument 2 (ID: INV-002):\n...",
            name="document_search", tool_call_id="s",
        ),
        reader_message("INV-002"),
        ToolMessage(content="Document with ID INV-999 not found.", name="document_reader", tool_call_id="x"),
        ToolMessage(content="TOTAL: 3", name="document_statistics", tool_call_id="t"),
    ]

    assert agent.document_ids_from_tools(messages) == ["INV-003", "INV-002"]


def search_message(*doc_ids, call_id="s"):
    content = "Found documents:\n\n" + "\n".join(f"Document {i} (ID: {doc_id}):" for i, doc_id in enumerate(doc_ids, 1))
    return ToolMessage(content=content, name="document_search", tool_call_id=call_id)


def test_select_sources_prefers_cited_documents():
    turn = [search_message("INV-003", "CON-001", "INV-002")]
    answer = AnswerResponse(question="q", answer="a", sources=["INV-003", "INV-002"], confidence=0.9)

    assert agent.select_sources(turn, answer) == ["INV-003", "INV-002"]


def test_select_sources_ignores_cited_documents_tools_never_found():
    turn = [search_message("CON-001", "INV-001"), reader_message("CON-001")]
    answer = AnswerResponse(question="q", answer="a", sources=["INV-999"], confidence=0.9)

    assert agent.select_sources(turn, answer) == ["CON-001"]


def test_select_sources_prefers_read_documents_over_search_results():
    turn = [search_message("CON-001", "INV-001", "INV-002"), reader_message("CON-001")]

    assert agent.select_sources(turn, None) == ["CON-001"]


def test_select_sources_uses_search_results_when_nothing_read():
    turn = [search_message("INV-003", "INV-002")]

    assert agent.select_sources(turn, None) == ["INV-003", "INV-002"]


def test_select_sources_uses_summary_document_ids():
    turn = [search_message("INV-001", "INV-002", "INV-003")]
    summary = SummarizationResponse(original_length=1, summary="s", key_points=[], document_ids=["INV-002"])

    assert agent.select_sources(turn, summary) == ["INV-002"]


def test_select_sources_handles_bad_summary_default():
    turn = [reader_message("INV-001")]
    summary = SummarizationResponse(original_length=1, summary="s", key_points=[])

    assert agent.select_sources(turn, summary) == ["INV-001"]


def test_update_memory_uses_cited_sources_from_current_response():
    llm = FakeLLM(document_ids=[])
    answer = AnswerResponse(question="q", answer="60 days", sources=["CON-001"], confidence=0.9)
    state = {
        "messages": [
            HumanMessage(content="Termination notice?"),
            search_message("CON-001", "INV-001", "INV-002"),
            AIMessage(content="60 days"),
        ],
        "current_response": {"structured_response": answer},
        "conversation_summary": "",
    }

    assert agent.update_memory(state, make_config(llm))["active_documents"] == ["CON-001"]


def test_update_memory_uses_documents_from_this_turn_not_llm_guess():
    llm = FakeLLM(document_ids=["INV-999"])
    state = {
        "messages": [
            HumanMessage(content="Tell me about CON-001"), reader_message("CON-001", "1"), AIMessage(content="A contract"),
            HumanMessage(content="And CLM-001?"), reader_message("CLM-001", "2"), AIMessage(content="A claim"),
        ],
        "conversation_summary": "",
    }

    assert agent.update_memory(state, make_config(llm))["active_documents"] == ["CLM-001"]


def test_update_memory_follow_up_keeps_only_previously_found_documents():
    llm = FakeLLM(document_ids=["INV-002", "INV-999"])
    state = {
        "messages": [
            HumanMessage(content="Show INV-002"), reader_message("INV-002"), AIMessage(content="$69,300"),
            HumanMessage(content="Double it"), AIMessage(content="$138,600"),
        ],
        "conversation_summary": "",
    }

    assert agent.update_memory(state, make_config(llm))["active_documents"] == ["INV-002"]


def test_update_memory_has_no_sources_for_missing_document():
    llm = FakeLLM(document_ids=["INV-999"])
    state = {
        "messages": [
            HumanMessage(content="Total of INV-999?"),
            ToolMessage(content="Document with ID INV-999 not found.", name="document_reader", tool_call_id="1"),
            AIMessage(content="Not found"),
        ],
        "conversation_summary": "",
    }

    assert agent.update_memory(state, make_config(llm))["active_documents"] == []


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


def fake_create_agent(monkeypatch, new_messages, structured_response=None):
    class FakeAgent:
        def invoke(self, payload):
            result = {"messages": payload["messages"] + new_messages}
            if structured_response is not None:
                result["structured_response"] = structured_response
            return result

    monkeypatch.setattr(agent, "create_agent", lambda **kwargs: FakeAgent())


def test_invoke_react_agent_counts_only_new_tool_calls(monkeypatch):
    history = [ToolMessage(content="old", name="document_reader", tool_call_id="old-1")]
    new = [ToolMessage(content="new", name="calculator", tool_call_id="new-1"), AIMessage(content="done")]
    fake_create_agent(monkeypatch, new)

    _, tools_used = agent.invoke_react_agent(AnswerResponse, history, llm=None, tools=[])

    assert tools_used == ["calculator"]


def test_invoke_react_agent_replaces_json_with_readable_answer(monkeypatch):
    structured = AnswerResponse(question="Who is the client?", answer="Healthcare Partners LLC", confidence=0.9)
    json_message = AIMessage(content=structured.model_dump_json(), id="final-id")
    fake_create_agent(monkeypatch, [json_message], structured)

    result, _ = agent.invoke_react_agent(AnswerResponse, [HumanMessage(content="hi")], llm=None, tools=[])

    assert result["messages"][-1].content == "Healthcare Partners LLC"
    assert result["messages"][-1].id == "final-id"
    assert result["structured_response"] is structured


def test_invoke_react_agent_keeps_message_without_structured_response(monkeypatch):
    fake_create_agent(monkeypatch, [AIMessage(content="plain answer")])

    result, _ = agent.invoke_react_agent(AnswerResponse, [], llm=None, tools=[])

    assert result["messages"][-1].content == "plain answer"


def test_format_summarization_response():
    response = SummarizationResponse(
        original_length=500, summary="A service agreement.", key_points=["12 months", "$180,000 total"],
    )

    assert agent.format_structured_response(response) == (
        "A service agreement.\n\nKey points:\n- 12 months\n- $180,000 total"
    )


@pytest.mark.parametrize("expression, expected", [
    ("15000 * 18", True),
    ("(214500 + 69300) / 2", True),
    ("$1,500.00 + $250", True),
    ("2 ** 10", True),
    ("(2) * (3)", True),
    ("N/A", False),
    ("", False),
    ("466250", False),
    ("Total Documents: 5, Total Amount: $466,250.00", False),
    ("The invoice with the highest amount is Invoice #12347 with an amount of $214,500.00.", False),
    ("max(22000, 69300, 214500)", False),
])
def test_is_math_expression(expression, expected):
    assert agent.is_math_expression(expression) is expected


def test_format_calculation_response_with_sentence_expression():
    response = CalculationResponse(
        expression="Total Documents: 5, Total Amount: $466,250.00",
        result=466250,
        explanation="The total value of the documents is $466,250.",
    )

    assert agent.format_structured_response(response) == "The total value of the documents is $466,250."


def test_format_calculation_response_without_calculation():
    response = CalculationResponse(
        expression="N/A", result=0, explanation="The document with ID INV-999 could not be found.",
    )

    assert agent.format_structured_response(response) == "The document with ID INV-999 could not be found."


def test_format_calculation_response():
    response = CalculationResponse(
        expression="22000 + 69300 + 214500", result=305800, explanation="Added the three invoice totals.", units="USD",
    )

    assert agent.format_structured_response(response, calculator_verified=True) == (
        "Added the three invoice totals.\n\nResult: 22000 + 69300 + 214500 = 305,800 USD"
    )


def test_format_calculation_response_hides_unverified_result():
    response = CalculationResponse(expression="15000 * 18", result=270000, explanation="Multiplied the fee by 18.")

    assert agent.format_structured_response(response) == "Multiplied the fee by 18."


@pytest.fixture
def calculator_tools(logger):
    from tools import create_calculator_tool
    return [create_calculator_tool(logger)]


def test_verify_with_calculator_confirms_correct_result(calculator_tools):
    response = CalculationResponse(expression="(1200 / 2450) * 100", result=48.9796, explanation="Divided.")

    verified_response, verified = agent.verify_with_calculator(response, calculator_tools)

    assert verified
    assert verified_response.result == pytest.approx(48.9795918367)
    assert verified_response.explanation == "Divided."


def test_verify_with_calculator_corrects_wrong_result(calculator_tools):
    response = CalculationResponse(expression="15000 * 18", result=250000, explanation="It is $250,000.")

    verified_response, verified = agent.verify_with_calculator(response, calculator_tools)

    assert verified
    assert verified_response.result == 270000
    assert "the calculator gives 270,000" in verified_response.explanation


@pytest.mark.parametrize("model_result", [48.98, 49.0, 48.9796, 48.979591836734695])
def test_verify_with_calculator_accepts_rounded_result(calculator_tools, model_result):
    response = CalculationResponse(expression="(1200 / 2450) * 100", result=model_result, explanation="Divided.")

    verified_response, verified = agent.verify_with_calculator(response, calculator_tools)

    assert verified
    assert verified_response.explanation == "Divided."


@pytest.mark.parametrize("expression", ["N/A", "Total Amount: $466,250.00", "5 / 0"])
def test_verify_with_calculator_rejects_non_math_and_errors(calculator_tools, expression):
    response = CalculationResponse(expression=expression, result=0, explanation="x")

    assert agent.verify_with_calculator(response, calculator_tools) == (response, False)


def test_verify_with_calculator_needs_calculator_tool():
    response = CalculationResponse(expression="2 + 2", result=4, explanation="x")

    assert agent.verify_with_calculator(response, []) == (response, False)


def test_invoke_react_agent_runs_calculator_when_model_skipped_it(monkeypatch, calculator_tools, logger):
    structured = CalculationResponse(expression="(1200 / 2450) * 100", result=48.98, explanation="About 49%.", units="%")
    fake_create_agent(monkeypatch, [AIMessage(content=structured.model_dump_json())], structured)

    result, tools_used = agent.invoke_react_agent(CalculationResponse, [], llm=None, tools=calculator_tools)

    assert tools_used == ["calculator"]
    assert [entry["tool_name"] for entry in logger.get_logs()] == ["calculator"]
    assert result["messages"][-1].content == "About 49%.\n\nResult: (1200 / 2450) * 100 = 48.9796 %"


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
