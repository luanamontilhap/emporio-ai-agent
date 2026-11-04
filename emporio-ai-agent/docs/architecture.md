# Architecture and implementation decisions

## Milestone 1 — deterministic consultation tools

### Goal

Provide reliable, testable access to the Empório da Música CSV snapshot before integrating an LLM. The LLM must not be responsible for calculating prices, interpreting stock quantities, joining customer/order records, or inventing operational data.

### Components

- `data_loader.py`: reads CSV files, normalizes whitespace and common encoding corruption, validates required columns, primary-key uniqueness, and foreign-key relationships. It also provides dataset diagnostics.
- `tools.py`: exposes deterministic operations as JSON-serializable dictionaries.
- `__main__.py`: lightweight CLI to exercise the tools manually.
- `tests/`: automated tests for core business rules and privacy checks.

### Data access strategy

Pandas is used because the supplied datasets are small CSV tables with explicit identifiers and relationships. The CSV data is not embedded in the prompt and does not require vector search. A database such as SQLite may become useful if the dataset grows or concurrent updates are introduced.

### Implemented tools

1. `search_products(query, category, max_price_brl, available_only, limit)`
2. `get_product_details(product_id)`
3. `get_product_price(product_id, payment_method)`
4. `get_order_details(order_id, customer_phone)`
5. `get_customer_orders(customer_phone)`
6. `get_store_information()`
7. `StoreData.diagnostics()`

### Business rules

- A product is available for sale only if its status is `active` and stock is greater than zero.
- Product search defaults to showing only products available for sale.
- Promotion rows with `is_active == 1` are treated as active. The promotions CSV does not provide start/end dates; therefore, calendar validity cannot be independently verified.
- If an active promotion exists, its discount is applied. The 5% PIX discount is not stacked on top of a promotion, following the policy manual.
- If no active promotion exists and payment method is PIX, the 5% PIX discount is applied to the list price.
- The recorded `orders.total_brl` is preserved as the official order total. A diagnostic compares it with the sum of order items but does not silently overwrite it.
- Order details are returned only after matching the supplied phone number with the registered customer's phone number. This is a prototype-only verification measure, not production authentication.
- The application identifies the CSVs as a static snapshot, not live operational data.

### Security and privacy

The order lookup avoids returning customer email, address, or unrelated customer details. Phone verification is not sufficient for a real deployment: use an authenticated session or one-time verification code (OTP), apply rate limiting, and avoid logging raw phone numbers.

### Known source-data ambiguities

- The manual lists `(67) 3341-4444` as the company telephone and `(67) 3321-4500` as the WhatsApp channel. Confirm the correct public channel before deployment.
- The manual's payment table says credit card up to 12 installments with a minimum installment of R$ 100, while section 3.1 defines different minimums for 1–3, 4–6, and 7–12 installments. No payment eligibility calculator is implemented until this conflict is clarified.
- The promotions file has an `is_active` flag but no effective dates. The flag is used as provided; it is not independently validated against today's date.
- The dataset's order totals for orders 3 and 20 differ from the sum of their listed items. The recorded order total is preserved and discrepancies are exposed by diagnostics.
- Customer, category, and order-item CSVs in this prototype were transcribed from tables supplied in the conversation. Compare them with original files if those files become available.

### Next milestone

Implement retrieval over `documents/politicas_da_loja.pdf`, then connect the deterministic tools and policy retriever to an LLM with native tool/function calling. Keep tool execution controlled and validate tool arguments before execution.


## Milestone 2 — policy-document retrieval (RAG)

### Goal

Retrieve policy evidence from the supplied Portuguese PDF while keeping retrieval separate from LLM answer generation.

### Pipeline

1. PyMuPDF extracts text and page boundaries from `documents/politicas_da_loja.pdf`.
2. Text is grouped by detected section headings and split into overlapping chunks.
3. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` encodes the chunks and each incoming question.
4. Cosine similarity ranks the chunks; the retriever returns the best passages with page, section, source, and score metadata.
5. `build_context()` formats the evidence for a future grounded LLM prompt.

### Why this implementation

The manual is a short, static, eight-page document in Portuguese. A multilingual sentence-transformer is more suitable for semantic question-to-passage matching than exact keyword matching alone. The embeddings are held in memory and compared with NumPy because a vector database would be unnecessary infrastructure for this small corpus. If the corpus grows, needs persistence, or becomes multi-tenant, a dedicated vector index can be reconsidered.

### Grounding and limitations

- Retrieved text is evidence, not an answer. The LLM integration must be instructed to answer only from retrieved passages, cite page numbers, and abstain/escalate when evidence is absent.
- The `min_score` value is a retrieval threshold heuristic, not a calibrated probability. It needs evaluation against representative Portuguese questions.
- The PDF includes conflicting installment minimum rules in section 3 and section 3.1. The agent must surface the conflict and avoid inventing a resolution.
- The document is version 2.1 from June 2025. It may not reflect later policy changes.
- This milestone does not yet connect an LLM or expose the retriever as a function-calling tool.

### Files

- `policy_rag.py`: extraction, chunking, embedding, retrieval, and context formatting.
- `tests/test_policy_rag.py`: tests use a deterministic fake encoder, so they do not download the model or depend on external services.


## Policy retrieval

The policy PDF uses subsection-aware chunking. Numbered headings are retained as metadata, chunks do not cross subsection boundaries, and a subsection may continue across PDF pages. Retrieval combines normalized multilingual embedding cosine similarity (70%) with BM25 lexical relevance (30%). The initial weighting is a transparent heuristic and should be evaluated on a labeled query set before production use. The retriever returns evidence and citations; it does not generate answers.


## Milestone 3 — LLM orchestration

### Goal

Connect the deterministic tools and the policy retriever to an LLM with native tool calling, while keeping every fact the customer sees traceable to a tool result.

### Decisions

- **OpenAI-compatible protocol, configured by environment variables.** The agent depends on a small `ChatClient` protocol; `OpenAICompatibleClient` implements it with the `openai` SDK and a configurable `base_url`, so a hosted API and a local server (for example Ollama) are interchangeable. `LLM_MODEL` has no default on purpose.
- **RAG is a tool.** `search_store_policy` is exposed like the other tools, so the model decides when the manual is needed and the same validation, tracing and evidence bookkeeping apply.
- **Evidence ledger per session.** Every number and page returned by a tool is recorded; the final answer is checked against that set (`validation.py`). Tool payloads are not replayed in later turns, only user/assistant text.
- **Fixed templates for refusals.** Out-of-scope, no-evidence, conflict and error replies are not generated by the model.
- **Privacy guards live in the dispatcher**, not in the prompt: phone digits must come from the customer's own messages, and repeated failed lookups lock the order tools.

### Control flow

`SupportAgent.ask` loops: call the LLM -> if it requests tools, execute them and continue -> if it writes text, apply the no-evidence gate and the validator -> accept, retry once with feedback, or abstain. Budgets (`AGENT_MAX_TOOL_ROUNDS`, `AGENT_MAX_VALIDATION_RETRIES`) bound the number of LLM calls.

### Testing strategy

A scripted LLM replays tool calls and answers, so the tests exercise the real tools, the real PDF/RAG pipeline (with the fake encoder), the dispatcher and the validator without network access. A local fake HTTP server verifies the real client against the Chat Completions wire format, including a full agent round trip. Live-model smoke tests are opt-in.

