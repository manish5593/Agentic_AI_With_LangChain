# Project Report: DocDacity Document Assistant

## 1. Summary

This project completes a document assistant that answers questions about, summarizes, and
performs calculations on five sample business documents: three invoices (INV-001, INV-002,
INV-003), a service agreement (CON-001) and an insurance claim (CLM-001).

All placeholder code in the starter project has been implemented. The assistant runs end to
end against GPT-4o, and 118 automated tests were written for it. The 113 offline tests pass.
The 5 tests that call the real model passed before the API budget ran out; the last change
(how sources are chosen) was checked with a real conversation but not with a full integration
run.

Testing with real conversations exposed several problems beyond the assigned tasks. These were
fixed and are described in section 5.

Environment: Python 3.12, LangChain 1.4.6, LangGraph 1.2.15, langchain-openai 1.7.2,
Pydantic 2.14, with GPT-4o reached through the Vocareum OpenAI endpoint.

## 2. Architecture

Every user message runs through one LangGraph graph:

```
                    ┌─> qa_agent ────────────┐
classify_intent ────┼─> summarization_agent ─┼─> update_memory ─> END
                    └─> calculation_agent ───┘
```

| Node | Role | Output |
|---|---|---|
| `classify_intent` | Asks the model to label the request | `UserIntent` (type, confidence, reasoning) and the next node |
| `qa_agent` | Answers a question using the document tools | `AnswerResponse` |
| `summarization_agent` | Summarizes one or more documents | `SummarizationResponse` |
| `calculation_agent` | Finds the numbers and computes with the calculator | `CalculationResponse` |
| `update_memory` | Updates the conversation summary and the documents in use | `UpdateMemoryResponse` |

The three agents share four tools: `document_search`, `document_reader`, `document_statistics`
and `calculator`.

State is saved with an `InMemorySaver` checkpointer under the session ID (`thread_id`). This is
what lets a later message such as "What is 15 percent of that amount?" refer back to an earlier
one.

## 3. Implementation

### 3.1 Schemas (`src/schemas.py`)

- `AnswerResponse` holds the question, the answer, the IDs of the documents used, a confidence
  score and a timestamp. `sources` defaults to an empty list and the timestamp fills itself in.
- `UserIntent` limits `intent_type` to `qa`, `summarization`, `calculation` or `unknown` using a
  `Literal` type, and limits `confidence` to 0–1.

Restricting these values in the schema means an invalid reply from the model is rejected by
Pydantic instead of being passed on. Every field has a description, because the model reads
those descriptions when filling in the structure.

### 3.2 Workflow (`src/agent.py`)

- **State.** `actions_taken` uses an `operator.add` reducer, so each node appends its name and a
  turn records its full path, e.g. `['classify_intent', 'calculation_agent', 'update_memory']`.
  `messages` uses `add_messages`, so the conversation grows across turns.
- **`classify_intent`** formats the classification prompt with the user's message and the chat
  history, calls the model with `with_structured_output(UserIntent)`, and maps the intent to a
  node. `unknown` falls back to the Q&A agent.
- **`summarization_agent` and `calculation_agent`** follow the pattern of the provided
  `qa_agent`, each with its own prompt and response schema.
- **`update_memory`** now receives the config so it can reach the model, and returns the
  summary, the active documents and `next_step = "end"`.
- **`create_workflow`** registers the five nodes, the conditional routing and the edges into
  `update_memory`, and compiles the graph with the checkpointer.

A bug in the starter file was also fixed: `should_continue` and `create_workflow` were indented
inside `update_memory`, so they could not be imported.

### 3.3 Assistant (`src/assistant.py`)

`process_message` passes the session ID as `thread_id`, plus the model and the tools, in the
graph config. The "no active session" check was moved before the config is built, so calling
it without a session gives a clear error instead of an `AttributeError`.

### 3.4 Prompts (`src/prompts.py`)

- The intent classification prompt gives three example requests for each category (`qa`,
  `summarization`, `calculation`, `unknown`). It sets confidence bands (0.9–1.0 clear, 0.7–0.89
  likely, 0.5–0.69 ambiguous, below 0.5 use `unknown`), says that requests needing arithmetic are
  calculations, and asks for one or two sentences of reasoning that name the deciding words.
- `get_chat_prompt_template` returns the Q&A, summarization or calculation system prompt for
  each intent, defaulting to Q&A.
- `CALCULATION_SYSTEM_PROMPT` tells the model to find and read the right document, work out the
  expression, and use the calculator for every calculation. It also says to show its steps,
  include units, cite document IDs and admit when numbers are missing.

### 3.5 Calculator tool (`src/tools.py`)

The calculator is a LangChain `@tool` that:

1. Removes `$` signs and thousands separators, so `$1,500.00 + $250` works.
2. Allows only digits, spaces, `.`, `+ - * / % **` and parentheses.
3. Rejects any exponent that is not a plain number, is larger than 100, or is nested inside
   another power.
4. Evaluates the expression with `eval()` and no access to Python built-ins.
5. Rounds away floating-point noise, so `0.1 + 0.2` gives `0.3`.
6. Logs every call with `ToolLogger`, and returns an error message instead of raising.

Step 3 was added after testing showed that `9**9**9` passed the character check and froze the
app.

## 4. Testing

| File | Tests | Covers |
|---|---|---|
| `test_schemas.py` | 12 | Defaults, allowed intents, confidence limits |
| `test_tools.py` | 36 | Calculator results, unsafe input, huge exponents, errors, logging; document reader |
| `test_prompts.py` | 9 | Prompt chosen for each intent; calculator rules in the calculation prompt; examples, confidence and reasoning rules in the classification prompt |
| `test_agent.py` | 56 | Routing, step order, memory across turns, separate sessions, source selection, calculator checks, formatting |
| `test_integration.py` | 5 | Real conversations: question, summary, calculation, follow-up, missing document |

The offline tests replace the model with a fake that returns fixed answers. They run in about
0.15 seconds and need no API key. The integration tests use `temperature=0` so their results
are repeatable, and are skipped when no API key is set.

### Results from real conversations

| Request | Intent | Answer | Correct |
|---|---|---|---|
| What is the combined total of all three invoices? | calculation | $305,800 (`22000 + 69300 + 214500`) | Yes |
| What is 15 percent of that claim amount? (follow-up on CLM-001) | calculation | $367.50 | Yes |
| What is the average of those invoice totals? (follow-up) | calculation | $141,900 | Yes |
| What percentage of the claim is the hospital visit? | calculation | 48.98% (`(1200 / 2450) * 100`) | Yes |
| How much tax was charged across all three invoices? | calculation | $27,800 | Yes |
| Which invoices are over $50,000? | qa | INV-003 and INV-002 | Yes |
| What is the termination notice period in the service agreement? | qa | 60 days' written notice | Yes |
| Summarize the service agreement | summarization | Parties, services, $15,000/month, $180,000 total, 12 months | Yes |
| What is the total of invoice INV-999? | calculation | Reports that the document could not be found | Yes |

### Runtime evidence

The assistant writes two kinds of files while it runs, and both are included in the repository:

- **`starter/sessions/`** holds 14 session files, one per run of `python main.py`. Each file
  stores the session ID, user ID, creation time and every turn of the conversation: the user's
  message, the messages exchanged with the model, the detected intent, the steps taken, the
  sources and the conversation summary.
- **`starter/logs/`** holds 14 tool logs (`tool_usage_<date>_<time>.json`), one per run. Each
  entry records the time, the tool called, its input and a short form of its output. For
  example, `tool_usage_20261015_045739.json` shows `document_search` and `document_reader` being
  used to find the invoices, then `calculator` evaluating `19500 + 6300 + 2000` = 27800.

All of them were produced by real conversations with GPT-4o on 15 October 2026. They cover every
request in the table above, including follow-ups such as "What is 15 percent of that claim
amount?" and the missing document INV-999.

## 5. Problems Found and Fixed

Each problem below was found by running the assistant or the tests, then fixed and covered by a
test.

| Problem | Cause | Fix |
|---|---|---|
| `prompts.py` would not import | `langchain.prompts` no longer exists in LangChain 1.x | Import from `langchain_core.prompts` |
| SOURCES never printed | `main.py` read `active_documents`; the assistant returns `sources` | Read the `sources` key |
| TOOLS USED grew every turn | Tool calls were counted across the whole chat history | Count only the current turn |
| Summary repeated the last answer | The model continued the chat instead of summarizing it | Pass a plain transcript and the previous summary, with a clearer prompt |
| Summary dropped earlier questions | The prompt capped it at 2–5 sentences | One bullet per question, never removing earlier ones |
| "Deserializing unregistered type" warnings | LangGraph did not trust the project's schema types | Register them with the checkpointer's serializer |
| Combined invoice total was $303,800 | INV-001 lists only a subtotal and tax, and the model used the subtotal | Prompt rule: use the total including tax; add subtotal and tax if no total is given |
| "There are no invoices in the system" | A keyword search for "invoices" matched nothing | Prompt rule: search by document type, and retry before giving up |
| `9**9**9` froze the app | Huge powers passed the character check | Exponent limit and no nested powers |
| Deprecation warnings | `create_react_agent` and Pydantic's `.dict()` are deprecated | Use `create_agent` and `model_dump()`, and turn `create_agent`'s JSON reply into readable text |
| `Result: N/A = 0` and sentences in the result line | The model filled the expression field with text | Show a result line only for real math |
| Results not always from the calculator | The model sometimes did the math itself | Run every calculation's expression through the calculator and show its result |
| SOURCES missing, wrong or too broad | The memory step guessed document IDs | Take sources from the tools: cited documents first, then documents read, then search results |

## 6. Known Limitations

- **Memory is lost on restart.** `InMemorySaver` keeps state in memory only. Session files are
  written to `starter/sessions/`, but `main.py` always starts a new session and the checkpointer
  is not reloaded from them. A persistent checkpointer such as `SqliteSaver` would fix this.
- **Follow-ups may show no sources.** A follow-up that reads no documents, e.g. "How much would
  18 months cost at that fee?", can show no SOURCES line.
- **The model does not always call the calculator for intermediate steps.** The final result is
  always checked by the calculator, but sub-steps such as adding a subtotal and tax may be done
  by the model.
- **Sample data.** INV-001 has no amount in its metadata, so `document_statistics` counts only 4
  of 5 documents and reports a total value of $466,250 without INV-001's $22,000.
- **Starter schema defaults.** `SummarizationResponse`, `UpdateMemoryResponse`, `DocumentChunk` and
  `SessionState` use `default_factory=lambda: list` (or `dict`), which gives the type itself as
  the default rather than an empty list. The source-selection code guards against this.
- **Session files grow quickly.** Each turn stores the full state, so a three-turn session is
  about 70 KB.
- **Early runs used earlier code.** The sessions and logs come from runs made while the fixes in
  section 5 were being applied, so the earliest ones show problems that were fixed later, such as
  the $303,800 total.

## 7. How to Run

From the repository root:

```bash
uv sync                                  # install dependencies
cp starter/.env.example starter/.env     # then add OPENAI_API_KEY to starter/.env
source .venv/bin/activate
cd starter && python main.py             # start the assistant
```

Tests, from the repository root:

```bash
uv run pytest -m "not integration"       # offline tests
uv run pytest -m integration             # real-model tests (use API budget)
```
