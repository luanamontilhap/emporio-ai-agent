# Empório da Música — AI Customer Support Agent

Prototype for the Artefact AI Engineer technical challenge.

## Current implementation

This first milestone implements deterministic Python tools over the supplied CSV snapshot:
- product search by text, category and maximum price;
- product detail and price lookup;
- active promotion handling;
- order lookup with a customer-phone verification step;
- customer's order listing;
- store information lookup;
- dataset diagnostics.

The second milestone adds a retrieval layer over `documents/politicas_da_loja.pdf`:
- PDF text extraction with PyMuPDF;
- heading-aware chunks that keep policy subsections separate, including sections spanning page breaks;
- multilingual sentence embeddings with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- hybrid ranking that combines dense cosine similarity (70%) with a lightweight BM25 lexical score (30%), preserving source page and section metadata;
- a context formatter for the future grounded LLM prompt.

The third milestone connects everything to an LLM through an OpenAI-compatible API (hosted or local):
- native tool calling over the deterministic tools and the policy retriever (`search_store_policy`);
- argument validation, privacy guards and a hand-off tool for out-of-scope or unsupported questions;
- a deterministic answer validator that rejects figures/citations without evidence, with one corrective retry;
- an interactive CLI: `python -m emporio_agent chat`.

## Requirements

- Python 3.11+
- VS Code recommended
- Internet access on first RAG run to download the embedding model (unless cached locally)

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m emporio_agent
```

To talk to the LLM-backed agent (after configuring `.env`, see below):

```powershell
$env:PYTHONPATH = "src"
python -m emporio_agent chat --trace
```

Example commands for the tool console (`python -m emporio_agent`):
- `products`
- `product 127`
- `price 127`
- `order 1 (67) 98432-1098`
- `info`
- `diagnostics`
- `policy Qual é o prazo para devolução por arrependimento?`
- `exit`

## Tests

```powershell
$env:PYTHONPATH = "src"
pytest -q
```

The default suite needs no network, API key or model download (the LLM is scripted and the embedding model is faked).
`tests/test_live_llm.py` runs against a real model and is skipped unless `RUN_LIVE_LLM_TESTS=1`.

## Data files

Place the source CSVs in `data/` using these names:
`products.csv`, `orders.csv`, `order_items.csv`, `customers.csv`, `categories.csv`, `promotions.csv`.

`products.csv`, `orders.csv`, and `promotions.csv` were copied from the provided attachments.
`customers.csv`, `categories.csv`, and `order_items.csv` were transcribed from tables pasted in the conversation. Their contents should be checked against the original source files if available.

## Important assumptions and limitations

1. The `is_active` field in `promotions.csv` is treated as the source of truth. The dataset has no start/end dates, so the tool cannot independently determine calendar validity.
2. PIX discount (5%) is applied only when no active promotion exists, following the policy that discounts are not cumulative.
3. A product is considered available for sale only when `status == "active"` and `stock_quantity > 0`.
4. The order's `total_brl` is treated as the official recorded total. The diagnostic function separately reports discrepancies against the sum of order items; it does not silently recalculate totals.
5. Phone matching is a prototype verification step, not production-grade authentication. A real deployment should use OTP or an authenticated customer session and avoid collecting full phone numbers in chat.
6. The CSVs are a static snapshot, not a live operational database. The agent must disclose this limitation.
7. Store phone numbers conflict in the manual: the company contact is listed as `(67) 3341-4444`, while WhatsApp is listed as `(67) 3321-4500`. The prototype returns the landline and flags the WhatsApp number for confirmation.
8. The manual contains different credit-card installment minimums in two sections. This requires clarification before implementing a payment-eligibility tool.

## Architecture decisions

- Pandas is used for deterministic structured-data operations.
- CSVs are not embedded into the LLM prompt and do not require vector retrieval.
- Business rules are implemented in Python functions so that they can be tested independently.
- The policy PDF is handled by a separate retrieval layer. It uses multilingual dense embeddings and cosine similarity. Because the manual is a small, static, eight-page document, embeddings are kept in memory and searched with a matrix multiplication; a vector database (FAISS/Chroma) would add operational complexity without meaningful benefit at this scale. Retrieved chunks retain page and section metadata so the final answer can cite the manual.
- Tool outputs are JSON-serializable dictionaries suitable for later function-calling integration.

## AI-assisted development

ChatGPT was used as a coding and design assistant to help structure the project, draft deterministic tools, and identify assumptions. Claude (Anthropic) was used as a coding assistant for the LLM-orchestration milestone (agent loop, validation, tests). The implementation should be reviewed and tested by the candidate, who remains responsible for the final code and technical decisions.

## Policy RAG design and limitations

- `policy_rag.py` extracts text from the supplied policy PDF and creates overlapping chunks inside numbered subsections (1,100 characters by default; 160-character overlap). Subsection boundaries are preserved, and content can continue across PDF page breaks.
- Retrieval combines multilingual dense embeddings (70%) and BM25 lexical matching (30%). The lexical component helps with exact policy terms and named locations; the dense component helps with paraphrases. These weights are initial heuristics, not empirically optimized values.
- Both ranking components run in memory. This is appropriate for one short PDF; a vector database is not necessary yet. The first `policy` command downloads the embedding model if it is not cached.
- Retrieval returns evidence passages, combined/dense/lexical relevance scores, subsection names, and original PDF page numbers. It does not generate a customer-facing answer by itself.
- The default minimum combined score is a tunable heuristic, not a calibrated confidence probability. Evaluate it with representative answerable and unanswerable questions before relying on it. Hybrid retrieval improves ranking but does not guarantee that the top passage is sufficient; the answer-generation layer must inspect evidence and abstain when it is incomplete.
- If no passage clears the threshold, the result explicitly reports that no evidence was retrieved. The future LLM prompt must then abstain or escalate instead of inventing a policy.
- The manual contains an inconsistency in credit-card installment minimums (payment table vs. section 3.1). Retrieval may surface both passages; the future agent must disclose the inconsistency and refer the case to store management rather than choosing one silently.

## LLM orchestration (milestone 3)

### Pipeline

```
question -> LLM (decides) -> tool calls -> tool results -> LLM (answers) -> validator -> customer
                 |                |                              |             |
                 |                +-- search_store_policy (RAG)  |             +-- reject + retry once, else abstain
                 +-- handoff_to_human (out of scope, no evidence, conflict, complaint) -> fixed template
```

| Module | Responsibility |
|---|---|
| `config.py` | Settings from environment variables / `.env` |
| `llm_client.py` | OpenAI-compatible Chat Completions adapter (`ChatClient` protocol) |
| `tool_registry.py` | Tool JSON schemas, argument validation, privacy guards, RAG tool, hand-off |
| `validation.py` | Deterministic grounding checks on the final answer |
| `prompts.py` | System prompt (pt-BR), hand-off templates, retry feedback |
| `agent.py` | The loop: `SupportAgent.ask(question, session)` |
| `session.py` | Per-conversation history and accumulated evidence |
| `chat_cli.py` | Interactive console |

### Configuration

All settings are environment variables (a `.env` file is read if present; real environment variables win). Copy `.env.example`.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `LLM_MODEL` | required | Model name served by that endpoint (no default, so it cannot go stale) |
| `LLM_API_KEY` | required, except for localhost URLs | |
| `LLM_TEMPERATURE` | unset (omitted) | Some models reject non-default values |
| `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` | 60, 2 | HTTP behavior |
| `AGENT_MAX_TOOL_ROUNDS` | 6 | Tool-calling rounds per question |
| `AGENT_MAX_VALIDATION_RETRIES` | 1 | Corrective retries after a rejected draft |
| `AGENT_MAX_HISTORY_MESSAGES` | 12 | User/assistant messages replayed per turn |
| `RAG_TOP_K`, `RAG_MIN_SCORE` | 4, 0.20 | Retrieval size and evidence threshold |

Hosted API: set `LLM_API_KEY` and `LLM_MODEL`. Ollama: `LLM_BASE_URL=http://localhost:11434/v1` and `LLM_MODEL=<a model with tool-calling support>`; no key needed.

### Safeguards

- **Tool execution is controlled.** Each call is parsed, validated against its schema (types, ranges, enums, unknown arguments rejected), executed by a deterministic function and serialized as JSON. Errors go back to the model as `{"ok": false, "error": ...}`; unknown tools and malformed JSON never execute anything.
- **Order privacy.** A phone number is accepted only if its digits appear in a message the customer typed in this conversation (the model cannot guess or enumerate numbers). Three failed lookups lock the order tools for the session. Phones are masked in traces, and tool payloads are not replayed in later turns.
- **No-evidence gate.** If the manual was searched and nothing relevant came back (and no other tool produced evidence), any free-form answer is replaced by a fixed "no reliable information" hand-off.
- **Answer validator.** Money, percentages, quantities/deadlines/installments, order/product identifiers and manual page citations in the final answer must appear in the tool/RAG results of the session. A rejected draft triggers one corrective retry; if it fails again the agent abstains.
- **Hand-off.** `handoff_to_human` ends the loop with a fixed pt-BR message (out of scope, accessories, insufficient evidence, manual conflict, complaint, needs a person). The contact data comes from `get_store_information`, never from the model.
- **Known source inconsistencies** (installment minimums; two WhatsApp/phone numbers) are attached as notes to the policy tool result so the model discloses them instead of choosing one.

### Limitations

- The validator catches fabricated **figures and citations**. It cannot verify prose that contains no numbers, and it accepts a number if it appears anywhere in the evidence, not necessarily in the field it is attributed to. Scope control relies on the system prompt and the model choosing `handoff_to_human`.
- The model must support native tool calling. Small local models may skip tools or produce malformed arguments; the pipeline then retries or abstains, but answer quality depends on the model. Run `tests/test_live_llm.py` against your chosen model.
- The model is told not to compute values (use `get_product_price`); a computed total or discount is rejected by the validator.
- Hand-off is a message only: no ticket or human-queue integration. There is no streaming and sessions live in memory.
- Pre-existing prototype caveats still apply: phone matching is not production authentication, and "order not found" vs "could not verify" reveals whether an order ID exists.
