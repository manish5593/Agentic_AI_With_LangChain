from datetime import datetime

import pytest
from pydantic import ValidationError

from schemas import AnswerResponse, UserIntent


def test_answer_response_defaults():
    response = AnswerResponse(question="What is the total?", answer="$100", confidence=0.8)

    assert response.sources == []
    assert isinstance(response.timestamp, datetime)


def test_answer_response_keeps_sources():
    response = AnswerResponse(question="q", answer="a", sources=["INV-001", "INV-002"], confidence=1.0)

    assert response.sources == ["INV-001", "INV-002"]


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_answer_response_rejects_confidence_out_of_range(confidence):
    with pytest.raises(ValidationError):
        AnswerResponse(question="q", answer="a", confidence=confidence)


@pytest.mark.parametrize("intent_type", ["qa", "summarization", "calculation", "unknown"])
def test_user_intent_accepts_valid_types(intent_type):
    intent = UserIntent(intent_type=intent_type, confidence=0.5, reasoning="because")

    assert intent.intent_type == intent_type


def test_user_intent_rejects_unknown_type():
    with pytest.raises(ValidationError):
        UserIntent(intent_type="chat", confidence=0.5, reasoning="because")


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_user_intent_rejects_confidence_out_of_range(confidence):
    with pytest.raises(ValidationError):
        UserIntent(intent_type="qa", confidence=confidence, reasoning="because")


def test_user_intent_requires_reasoning():
    with pytest.raises(ValidationError):
        UserIntent(intent_type="qa", confidence=0.5)
