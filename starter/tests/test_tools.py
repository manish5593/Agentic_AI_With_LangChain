import json

import pytest

from tools import create_calculator_tool, create_document_reader_tool, get_all_tools


@pytest.fixture
def calculator(logger):
    return create_calculator_tool(logger)


@pytest.mark.parametrize("expression, expected", [
    ("2 + 2", "2 + 2 = 4"),
    ("(1500 + 2750.50) * 0.08", "(1500 + 2750.50) * 0.08 = 340.04"),
    ("10 / 4", "10 / 4 = 2.5"),
    ("2 ** 10", "2 ** 10 = 1024"),
    ("17 % 5", "17 % 5 = 2"),
    ("0.1 + 0.2", "0.1 + 0.2 = 0.3"),
    ("-5 + 3", "-5 + 3 = -2"),
    ("22000 + 69300 + 214500", "22000 + 69300 + 214500 = 305800"),
])
def test_calculator_evaluates_expressions(calculator, expression, expected):
    assert calculator.invoke({"expression": expression}) == expected


def test_calculator_strips_currency_formatting(calculator):
    assert calculator.invoke({"expression": "$1,500.00 + $250"}) == "$1,500.00 + $250 = 1750"


def test_calculator_handles_division_by_zero(calculator):
    assert calculator.invoke({"expression": "5 / 0"}) == "Error calculating expression: division by zero"


@pytest.mark.parametrize("expression", [
    "__import__('os').system('ls')",
    "open('secrets.txt')",
    "abs(-5)",
    "2 + x",
    "1e9",
    "",
])
def test_calculator_rejects_unsafe_or_invalid_characters(calculator, expression):
    result = calculator.invoke({"expression": expression})

    assert result.startswith("Error calculating expression")
    assert "Invalid characters" in result


@pytest.mark.parametrize("expression", ["1 +", "(2 + 3", "* 4"])
def test_calculator_rejects_malformed_expressions(calculator, expression):
    assert "is not a valid math expression" in calculator.invoke({"expression": expression})


@pytest.mark.parametrize("expression, expected", [
    ("2 ** 100", "2 ** 100 = 1267650600228229401496703205376"),
    ("2 ** -2", "2 ** -2 = 0.25"),
    ("1.05 ** 12", "1.05 ** 12 = 1.795856326"),
    ("(2 ** 3) * (3 ** 2)", "(2 ** 3) * (3 ** 2) = 72"),
])
def test_calculator_allows_reasonable_powers(calculator, expression, expected):
    assert calculator.invoke({"expression": expression}) == expected


@pytest.mark.parametrize("expression, message", [
    ("9 ** 9 ** 9", "Exponents must be plain numbers"),
    ("(10 ** 10) ** (10 ** 10)", "Exponents must be plain numbers"),
    ("2 ** (50 + 60)", "Exponents must be plain numbers"),
    ("2 ** 101", "Exponent too large"),
    ("10 ** 1000000", "Exponent too large"),
    ("2 ** -101", "Exponent too large"),
    ("(2 ** 50) ** 2", "Nested powers are not allowed"),
])
def test_calculator_rejects_dangerous_powers(calculator, expression, message):
    result = calculator.invoke({"expression": expression})

    assert result.startswith("Error calculating expression")
    assert message in result


def test_calculator_rejects_non_numeric_result(calculator):
    assert "did not evaluate to a number" in calculator.invoke({"expression": "()"})


def test_calculator_always_returns_a_string(calculator):
    for expression in ["1 + 1", "5 / 0", "bad"]:
        assert isinstance(calculator.invoke({"expression": expression}), str)


def test_calculator_logs_successes_and_errors(calculator, logger):
    calculator.invoke({"expression": "1 + 1"})
    calculator.invoke({"expression": "5 / 0"})

    logs = logger.get_logs()
    assert [entry["tool_name"] for entry in logs] == ["calculator", "calculator"]
    assert logs[0]["input"] == {"expression": "1 + 1"}
    assert "2" in logs[0]["output"]
    assert "division by zero" in logs[1]["output"]

    with open(logger.log_file) as f:
        assert len(json.load(f)) == 2


def test_document_reader_finds_document(retriever, logger):
    reader = create_document_reader_tool(retriever, logger)

    result = reader.invoke({"doc_id": "INV-002"})

    assert result.startswith("Document INV-002:")
    assert "$69,300.00" in result


def test_document_reader_handles_missing_document(retriever, logger):
    reader = create_document_reader_tool(retriever, logger)

    assert reader.invoke({"doc_id": "NOPE-999"}) == "Document with ID NOPE-999 not found."


def test_get_all_tools_includes_calculator(retriever, logger):
    names = [t.name for t in get_all_tools(retriever, logger)]

    assert names == ["calculator", "document_search", "document_reader", "document_statistics"]
