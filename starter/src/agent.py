from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
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


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> (
Dict[str, Any], List[str]):
    llm_with_tools = llm.bind_tools(
        tools
    )

    agent = create_react_agent(
        model=llm_with_tools,  # Use the bound model
        tools=tools,
        response_format=response_schema,
    )

    result = agent.invoke({"messages": messages})
    # Only count tool calls made in this turn, not those already in the chat history
    new_messages = result.get("messages", [])[len(messages):]
    tools_used = [t.name for t in new_messages if isinstance(t, ToolMessage)]

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
    return {
        "conversation_summary": response.summary,
        "active_documents": response.document_ids,
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
