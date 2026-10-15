import os
import sys

import pytest
from langchain_core.messages import AIMessage, ToolMessage

# The src modules import each other as top-level modules (e.g. `from schemas import ...`)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schemas import UserIntent, UpdateMemoryResponse
from tools import ToolLogger
from retrieval import SimulatedRetriever


class FakeLLM:
    """Stand-in LLM that returns a fixed intent and memory summary, so tests need no API calls."""

    def __init__(self, intent_type="qa", summary="summary", document_ids=None):
        self.intent_type = intent_type
        self.summary = summary
        self.document_ids = document_ids or ["INV-001"]
        self.prompts = []

    def with_structured_output(self, schema):
        llm = self

        class StructuredLLM:
            def invoke(self, prompt):
                llm.prompts.append((schema, prompt))
                if schema is UserIntent:
                    return UserIntent(intent_type=llm.intent_type, confidence=0.9, reasoning="test")
                if schema is UpdateMemoryResponse:
                    return UpdateMemoryResponse(summary=llm.summary, document_ids=llm.document_ids)
                raise AssertionError(f"Unexpected schema {schema}")

        return StructuredLLM()


@pytest.fixture
def logger(tmp_path):
    return ToolLogger(logs_dir=str(tmp_path / "logs"))


@pytest.fixture
def retriever():
    return SimulatedRetriever()


@pytest.fixture
def fake_react_agent(monkeypatch):
    """Replace the ReAct agent with one that answers with the schema name and a single tool call."""
    import agent

    calls = []

    def fake_invoke_react_agent(response_schema, messages, llm, tools):
        calls.append(response_schema)
        reply = [
            ToolMessage(content="ok", name="document_reader", tool_call_id="call-1"),
            AIMessage(content=f"{response_schema.__name__} answer"),
        ]
        return {"messages": messages + reply}, ["document_reader"]

    monkeypatch.setattr(agent, "invoke_react_agent", fake_invoke_react_agent)
    return calls
