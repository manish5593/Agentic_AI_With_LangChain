# DocDacity Document Assistant

DocDacity is a command-line assistant that lets you chat with a small collection of business
documents: three invoices, a service agreement and an insurance claim. You type a request in
plain English, and the assistant works out what kind of request it is, looks up the documents
it needs, and replies.

It handles three kinds of requests:

- **Questions**, such as "What are the payment terms on INV-003?"
- **Summaries**, such as "Summarize the insurance claim CLM-001"
- **Calculations**, such as "How much tax was charged across all three invoices?"

Behind the scenes, a LangGraph workflow first classifies each message, then hands it to a
specialist agent for that kind of request. The agents use tools to search documents, read
them and do arithmetic, and a final step updates a running summary of the conversation. The
assistant remembers earlier messages, so follow-ups like "What is 15 percent of that amount?"
work. Every reply also shows the detected intent, the documents used, the tools called and
the conversation summary.

## Getting Started

These steps get a copy of the project running on your own computer. You need a terminal,
Python, and an OpenAI API key.

### Dependencies

```
Python 3.12 or newer
uv (recommended) or pip, to install packages
An OpenAI API key for the Vocareum endpoint (a key that starts with voc-)

Python packages, installed in step 2 below:
  langchain, langchain-core, langchain-openai   LLM framework and OpenAI connection
  langgraph                                      Multi-agent workflow and memory
  pydantic                                       Structured responses
  python-dotenv                                  Reads the API key from a .env file
  print-color                                    Colored terminal output
  pytest                                         Runs the automated tests
```

### Installation

1. Clone the repository and move into it.
2. Install the packages into a virtual environment.
3. Create a `.env` file in the `starter` folder that holds your API key.
4. Activate the environment and start the assistant from the `starter` folder.

```bash
# 1. Get the code
git clone https://github.com/manish5593/Agentic_AI_With_LangChain.git
cd Agentic_AI_With_LangChain

# 2. Install packages (creates a .venv folder)
uv sync
#    Without uv:
#    python -m venv .venv && source .venv/bin/activate
#    pip install -r starter/requirements.txt pytest

# 3. Add your API key
cp starter/.env.example starter/.env
#    Open starter/.env and replace the placeholder after OPENAI_API_KEY= with your key

# 4. Run the assistant
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
cd starter
python main.py
```

When it starts, press Enter to use the default user ID, then type a message. Type `/docs` to
list the documents, `/help` to see the commands, and `/quit` to exit. A short session looks
like this:

```
Enter Message: How much tax was charged across all three invoices?

🤖 Assistant: The tax on each invoice is INV-001: $2,000, INV-002: $6,300 and
INV-003: $19,500, so the total tax is $27,800.

Result: 2000 + 6300 + 19500 = 27,800 USD

INTENT: calculation
SOURCES: INV-003, INV-002, INV-001
TOOLS USED: document_search, document_reader, document_reader, document_reader, calculator
```

Keep the key only in `starter/.env`. That file is ignored by Git, while `.env.example` is
committed, so a key placed there would end up on GitHub.

## Testing

The automated tests use pytest and are in `starter/tests`. Run them from the repository root
(the folder that contains `pyproject.toml`):

1. Run the offline tests. They replace the language model with a fake one, so they need no
   API key, cost nothing and finish in under a second:

   ```bash
   uv run pytest -m "not integration"
   ```

2. Run the integration tests. They send real requests to the model, so they need the key in
   `starter/.env` and use some of your API budget. Without a key they are skipped:

   ```bash
   uv run pytest -m integration
   ```

3. Or run everything at once with `uv run pytest`.

If you installed with pip instead of uv, activate `.venv` and use `pytest` in place of
`uv run pytest`.

### Break Down Tests

The tests are split into one file per part of the project:

- **`test_schemas.py`** checks the data models the model must fill in. For example, an intent
  can only be one of `qa`, `summarization`, `calculation` or `unknown`, and a confidence score
  must be between 0 and 1. This matters because these models are what stop a bad model reply
  from flowing through the rest of the app.
- **`test_tools.py`** checks the tools the agents call. Most of it covers the calculator:
  correct results, handling `$` signs and thousands separators, clear error messages, logging,
  and rejecting input that could run code or freeze the app. Since the calculator uses
  Python's `eval()`, these safety checks are the most important tests in the project.
- **`test_prompts.py`** checks that each kind of request gets its own instructions and that the
  calculation instructions require the calculator tool. If the wrong instructions were sent,
  the agents would behave in the wrong way without any error.
- **`test_agent.py`** checks the workflow itself, using a fake model: that each kind of request
  reaches the right agent, that steps run in order, that memory carries over between messages
  and stays separate between sessions, that the listed sources are documents the tools really
  found, and that every displayed calculation result comes from the calculator.
- **`test_integration.py`** has a conversation with the real model: a question, a summary, a
  calculation, a follow-up that relies on memory, and a request for a document that doesn't
  exist. These catch problems the fake model can't, such as unclear prompts.

Examples of what the tests check:

```python
# Code hidden in a calculation is refused, not run
calculator.invoke({"expression": "__import__('os').system('ls')"})
# -> "Error calculating expression: Invalid characters in expression. ..."

# A calculation that would take forever is blocked
calculator.invoke({"expression": "9 ** 9 ** 9"})
# -> "Error calculating expression: Exponents must be plain numbers, e.g. 2 ** 10."

# A calculation request passes through these steps, in this order
# ['classify_intent', 'calculation_agent', 'update_memory']

# Asking about a document that doesn't exist gives a clear answer, not a made-up one
assistant.process_message("What is the total of invoice INV-999?")
# -> "The document with ID INV-999 could not be found ..."
```

## Project Instructions

To complete the project, students deliver a working assistant built from the provided starter
code in `starter/`. The finished project must:

1. **Define structured responses.** Write the Pydantic models for a question's answer
   (including the sources used and a confidence score) and for the detected intent, so the
   model's replies have a fixed, checkable shape.
2. **Build the workflow.** Complete the LangGraph graph so that every message is classified,
   sent to the Q&A, summarization or calculation agent, and then passed to a memory step. The
   intent must decide the route, and unclear requests must fall back to Q&A.
3. **Finish the specialist agents.** Write the summarization and calculation agents in the
   same style as the provided Q&A agent, each with its own instructions and response model.
4. **Add memory.** Keep a running conversation summary and the list of documents in use, track
   which steps ran in each turn, and save state between messages with a checkpointer, so that
   follow-up questions work within a session.
5. **Write the prompts.** Give each kind of request suitable instructions. The calculation
   instructions must make the model read the needed document and use the calculator for every
   calculation, however simple.
6. **Write the calculator tool.** It must check its input before evaluating it, return results
   as text, log each use, and return an error message instead of crashing.
7. **Test the result.** Check the individual parts, the full workflow with different requests,
   and edge cases such as bad input and missing documents.

The assistant works when it can classify requests correctly, find and cite the right
documents, answer questions, write summaries, do calculations from document data, and keep
track of the conversation across messages.

## Built With

* [Python](https://www.python.org/) - Programming language for the whole project
* [LangChain](https://www.langchain.com/) - Connects to the language model and provides tools, prompt templates and structured output
* [LangGraph](https://www.langchain.com/langgraph) - Builds the multi-agent workflow and saves conversation state between messages
* [OpenAI GPT-4o](https://platform.openai.com/docs/models) - The language model behind every agent, accessed through the Vocareum endpoint
* [Pydantic](https://docs.pydantic.dev/) - Defines and validates the structured responses
* [python-dotenv](https://github.com/theskumar/python-dotenv) - Loads the API key and settings from `starter/.env`
* [print-color](https://pypi.org/project/print-color/) - Prints colored output in the terminal
* [pytest](https://docs.pytest.org/) - Runs the automated tests
* [uv](https://docs.astral.sh/uv/) - Creates the virtual environment and installs the exact package versions in `uv.lock`

## License
[License](../LICENSE.md)
