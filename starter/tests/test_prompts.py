import pytest

from prompts import (
    CALCULATION_SYSTEM_PROMPT, QA_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT,
    get_chat_prompt_template, get_intent_classification_prompt,
)


@pytest.mark.parametrize("intent_type, expected_prompt", [
    ("qa", QA_SYSTEM_PROMPT),
    ("summarization", SUMMARIZATION_SYSTEM_PROMPT),
    ("calculation", CALCULATION_SYSTEM_PROMPT),
    ("unknown", QA_SYSTEM_PROMPT),
    ("something-else", QA_SYSTEM_PROMPT),
])
def test_chat_prompt_uses_system_prompt_for_intent(intent_type, expected_prompt):
    messages = get_chat_prompt_template(intent_type).invoke({
        "input": "hello",
        "chat_history": [],
    }).to_messages()

    assert messages[0].content == expected_prompt
    assert messages[-1].content == "hello"


def test_calculation_prompt_requires_calculator_and_document_reader():
    prompt = CALCULATION_SYSTEM_PROMPT.lower()

    assert prompt.strip()
    assert "calculator tool" in prompt
    assert "document reader tool" in prompt
    assert "always use the calculator tool" in prompt


def test_intent_classification_prompt_includes_inputs():
    prompt = get_intent_classification_prompt().format(
        user_input="Sum all invoices",
        conversation_history="human: hi",
    )

    assert "Sum all invoices" in prompt
    assert "human: hi" in prompt
