"""
End-to-end tests that call the real LLM through DocumentAssistant.
They are skipped unless OPENAI_API_KEY is set (e.g. in starter/.env).
Run only these with: pytest -m integration
"""
import os
import re

import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set"),
]


@pytest.fixture
def assistant(tmp_path, monkeypatch):
    from assistant import DocumentAssistant

    # Keep logs and sessions written by the assistant out of the project folder
    monkeypatch.chdir(tmp_path)
    assistant = DocumentAssistant(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name=os.getenv("MODEL_NAME", "gpt-4o"),
        temperature=0,
        session_storage_path=str(tmp_path / "sessions"),
    )
    assistant.start_session("test_user")
    return assistant


def test_question_about_a_document(assistant):
    result = assistant.process_message("Who is the client on the service agreement CON-001?")

    assert result["success"], result.get("error")
    assert result["intent"]["intent_type"] == "qa"
    assert "Healthcare Partners" in result["response"]
    assert "CON-001" in result["sources"]
    assert result["actions_taken"] == ["classify_intent", "qa_agent", "update_memory"]


def test_summarization_request(assistant):
    result = assistant.process_message("Summarize the insurance claim CLM-001")

    assert result["success"], result.get("error")
    assert result["intent"]["intent_type"] == "summarization"
    assert "2,450" in result["response"]
    assert "CLM-001" in result["sources"]


def test_calculation_uses_calculator(assistant):
    result = assistant.process_message("What is the combined total of all three invoices?")

    assert result["success"], result.get("error")
    assert result["intent"]["intent_type"] == "calculation"
    assert "calculator" in result["tools_used"]
    assert "305,800" in result["response"]


def test_follow_up_uses_conversation_memory(assistant):
    first = assistant.process_message("What is the claim amount of CLM-001?")
    follow_up = assistant.process_message("What is 15 percent of that amount?")

    assert first["success"] and follow_up["success"]
    assert "367.5" in follow_up["response"]
    assert "calculator" in follow_up["tools_used"]
    assert "CLM-001" in follow_up["summary"]


def test_missing_document_is_reported(assistant):
    result = assistant.process_message("What is the total of invoice INV-999?")

    assert result["success"], result.get("error")
    assert re.search(
        r"not (be )?found|(could ?n[o']t|unable to) (be )?(find|found|locate)|does ?n[o']t exist|no (such )?invoice",
        result["response"],
        re.IGNORECASE,
    ), result["response"]
