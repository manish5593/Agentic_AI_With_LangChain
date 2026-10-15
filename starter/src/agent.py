from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.prebuilt import tools_condition, ToolNode
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import math
import re
import operator
from schemas import (
    UserIntent, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
from prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT


class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]

    actions_taken: Annotated[List[str], operator.add]


def is_math_expression(expression: str) -> bool:
    """True if the expression is real math (numbers and operators), not a sentence or 'N/A'."""
    cleaned = re.sub(r"(?<=\d),(?=\d{3})", "", expression.replace("$", ""))
    return bool(
        re.fullmatch(r"[\d\s.+\-*/%()]+", cleaned)
        and re.search(r"\d\s*\)*\s*(\*\*|[-+*/%])\s*\(*\s*\d", cleaned)
    )


def verify_with_calculator(response: CalculationResponse, tools) -> (CalculationResponse, bool):
    """
    Run the response's expression through the calculator tool, so the result shown to the
    user always comes from the calculator even if the model did the math itself.
    Returns the response (with the calculator's result) and whether it was verified.
    """
    calculator = next((t for t in tools or [] if t.name == "calculator"), None)
    if calculator is None or not is_math_expression(response.expression):
        return response, False

    output = calculator.invoke({"expression": response.expression})
    match = re.search(r" = (-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$", output)
    if not match:
        return response, False

    value = float(match.group(1))
    explanation = response.explanation
    # The model's figure counts as matching if it is the calculator's value rounded (e.g. 48.98)
    model_text = repr(float(response.result))
    decimals = len(model_text.split(".")[1]) if "." in model_text and "e" not in model_text else 10
    if not (math.isclose(value, response.result, rel_tol=1e-6, abs_tol=1e-6)
            or round(value, decimals) == response.result):
        explanation += (
            f"\n\nNote: the calculator gives {value:,g} for this expression, "
            f"which differs from the figure above."
        )
    return response.model_copy(update={"result": value, "explanation": explanation}), True


def format_structured_response(response: BaseModel, calculator_verified: bool = False) -> str:
    """Turn an agent's structured response into readable text for the user."""
    if isinstance(response, AnswerResponse):
        return response.answer
    if isinstance(response, SummarizationResponse):
        key_points = "\n".join(f"- {point}" for point in response.key_points)
        return f"{response.summary}\n\nKey points:\n{key_points}" if key_points else response.summary
    if isinstance(response, CalculationResponse):
        # Only show a result line the calculator produced; skip it when nothing was calculated
        # (e.g. the document was not found, or the expression field holds a sentence)
        if not calculator_verified:
            return response.explanation
        units = f" {response.units}" if response.units else ""
        return f"{response.explanation}\n\nResult: {response.expression} = {response.result:,g}{units}"
    return str(response)


def document_ids_from_tools(messages: List[BaseMessage], search_results: bool = True) -> List[str]:
    """
    IDs of the documents that document_reader read or, unless search_results is False,
    that document_search returned, in order.
    """
    doc_ids = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        if message.name == "document_reader":
            doc_ids += re.findall(r"^Document ([\w-]+):", message.content)
        elif message.name == "document_search" and search_results:
            doc_ids += re.findall(r"\(ID: ([\w-]+)\)", message.content)
    return list(dict.fromkeys(doc_ids))


def cited_document_ids(structured_response) -> List[str]:
    """Document IDs an agent's structured response says it used, if its schema has them."""
    if isinstance(structured_response, AnswerResponse):
        doc_ids = structured_response.sources
    elif isinstance(structured_response, SummarizationResponse):
        doc_ids = structured_response.document_ids
    else:
        return []
    # Guard against non-list defaults, e.g. SummarizationResponse's `lambda: list` default
    return [doc_id for doc_id in doc_ids if isinstance(doc_id, str)] if isinstance(doc_ids, list) else []


def select_sources(turn_messages: List[BaseMessage], structured_response) -> List[str]:
    """
    The documents an answer actually used, preferring in order: the documents the answer
    cites (if the tools found them), the documents read in full, then all search results.
    """
    found = document_ids_from_tools(turn_messages)
    cited = [doc_id for doc_id in cited_document_ids(structured_response) if doc_id in found]
    if cited:
        return list(dict.fromkeys(cited))
    return document_ids_from_tools(turn_messages, search_results=False) or found


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> (
Dict[str, Any], List[str]):
    agent = create_agent(
        model=llm,
        tools=tools,
        response_format=response_schema,
    )

    result = agent.invoke({"messages": messages})
    # Only count tool calls made in this turn, not those already in the chat history
    new_messages = result.get("messages", [])[len(messages):]
    tools_used = [t.name for t in new_messages if isinstance(t, ToolMessage)]

    # create_agent ends with the structured response as a JSON message, so replace it
    # with readable text (keeping its id) for the user and the conversation history
    structured_response = result.get("structured_response")
    calculator_verified = False
    if isinstance(structured_response, CalculationResponse):
        structured_response, calculator_verified = verify_with_calculator(structured_response, tools)
        if calculator_verified:
            result["structured_response"] = structured_response
            if "calculator" not in tools_used:
                tools_used.append("calculator")

    if structured_response is not None and isinstance(result["messages"][-1], AIMessage):
        final_message = result["messages"][-1]
        result["messages"][-1] = AIMessage(
            content=format_structured_response(structured_response, calculator_verified),
            id=final_message.id,
        )

    return result, tools_used


def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable").get("llm")
    history = state.get("messages", [])

    structured_llm = llm.with_structured_output(UserIntent)

    conversation_history = "\n".join(
        f"{message.type}: {message.content}" for message in history if message.content
    ) or "No previous conversation."
    prompt = get_intent_classification_prompt().format(
        user_input=state["user_input"],
        conversation_history=conversation_history,
    )

    intent = structured_llm.invoke(prompt)

    next_step = "qa_agent"
    if intent.intent_type == "summarization":
        next_step = "summarization_agent"
    elif intent.intent_type == "calculation":
        next_step = "calculation_agent"

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent,
        "next_step": next_step,
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("qa")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("summarization")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(SummarizationResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["summarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("calculation")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(CalculationResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def update_memory(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """

    llm = config.get("configurable").get("llm")

    # Pass the conversation as a transcript so the LLM summarizes it instead of
    # continuing the conversation from the assistant's last reply
    transcript = "\n\n".join(
        f"{'User' if message.type == 'human' else 'Assistant'}: {message.content}"
        for message in state.get("messages", [])
        if message.type in ("human", "ai") and message.content
    )

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        ("human", "Previous summary:\n{previous_summary}\n\nConversation transcript:\n{transcript}"),
    ]).invoke({
        "previous_summary": state.get("conversation_summary") or "None",
        "transcript": transcript,
    })

    structured_llm = llm.with_structured_output(UpdateMemoryResponse)

    response = structured_llm.invoke(prompt_with_history)

    # Take the sources from the tools rather than trusting the LLM's guess: the documents
    # this turn's answer used, or for a follow-up that used no document tools, the
    # LLM's IDs limited to documents the tools have found earlier in the conversation
    messages = state.get("messages", [])
    last_user_message = max(
        (i for i, message in enumerate(messages) if isinstance(message, HumanMessage)), default=-1
    )
    structured_response = (state.get("current_response") or {}).get("structured_response")
    active_documents = select_sources(messages[last_user_message + 1:], structured_response)
    if not active_documents:
        known_documents = set(document_ids_from_tools(messages))
        active_documents = [doc_id for doc_id in response.document_ids if doc_id in known_documents]

    return {
        "conversation_summary": response.summary,
        "active_documents": active_documents,
        "actions_taken": ["update_memory"],
        "next_step": "end",
    }


def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")


def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "end": END
        }
    )

    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")

    workflow.add_edge("update_memory", END)

    # Allow the checkpointer to restore our own schema objects stored in state
    serde = JsonPlusSerializer(allowed_msgpack_modules=[
        ("schemas", "UserIntent"),
        ("schemas", "AnswerResponse"),
        ("schemas", "SummarizationResponse"),
        ("schemas", "CalculationResponse"),
    ])
    return workflow.compile(checkpointer=InMemorySaver(serde=serde))
