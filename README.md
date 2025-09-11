# Chatbot Classification (Azure OpenAI + PostgreSQL)

This project classifies chatbot **sessions** and **messages** using **Azure OpenAI**, stores results in **PostgreSQL**, and pulls agent instructions from **Azure Blob Storage**.  
It now runs through a **unified pipeline** that can perform session-level and/or message-level classification **in one pass** to avoid double fetching.

## 1) Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ````

2. Create `.env.local` (or copy `.env.example`) and fill in your values:

   ```env
   CLIENT_ENDPOINT=
   CLIENT_API_KEY=
   CLIENT_API_VERSION=2024-10-21
   ASSISTANT_MODEL=gpt-4o-mini

   AZURE_STORAGE_CONNECTION_STRING=
   AZURE_STORAGE_CONTAINER=chatbot-prompts
   AZURE_STORAGE_BLOB_MESSAGE_INSTRUCTIONS=message_instructions.txt
   AZURE_STORAGE_BLOB_SESSION_INSTRUCTIONS=session_instructions.txt

   DATABASE_URL=postgresql://pgadmin:<PW>@pgserver2-<NAME>-staging.postgres.database.azure.com/chatbot_db?sslmode=require
   ```

3. Edit `config/categories.yaml` to match your taxonomy.

4. Create two blobs in your Azure Storage container with your agent instructions:

   * `message_instructions.txt` – rules for classifying individual messages and the required JSON schema.
   * `session_instructions.txt` – rules for classifying entire sessions.

   *Tip:* Include explicit schema and the label list in each instruction blob. The code also passes the category list at runtime.

5. (Optional) Create result tables manually, or let the app create them on first run:

   ```bash
   psql "$DATABASE_URL" -f sql/create_result_tables.sql
   ```

## 2) Run (Unified Pipeline)

The CLI wraps a single `run_pipeline(...)` that can do session-level and/or message-level classification in one pass.

```bash
# Both session and message classification (default)
python -m src.main

# Only session classification
python -m src.main --no-messages

# Only message classification (user+assistant), with per-session batches of 20
python -m src.main --no-session --roles user assistant --per-session-message-batch-size 20

# Incremental since a timestamp; cap to 100 sessions
python -m src.main --since 2025-01-01T00:00:00Z --limit 100

# Force reclassifying messages even if they already have rows in message_classification
python -m src.main --reclassify-existing-messages

# Increase logging verbosity
python -m src.main --log-level DEBUG
```

### CLI flags

* `--roles ...`
  Roles to include when building transcripts and selecting messages (default: `user`).
  Examples: `--roles user assistant` or `--roles user assistant tool`.

* `--since <ISO>`
  Lower bound timestamp for incremental processing (e.g., `2025-01-01T00:00:00Z`).

* `--limit <N>`
  Maximum number of **sessions** to process.

* `--no-session` / `--no-messages`
  Disable either classifier. By default, **both** run.

* `--reclassify-existing-messages`
  Re-run message classification even if a row already exists in `message_classification`.

* `--per-session-message-batch-size <N>`
  Split each session’s messages into batches of size `N` per model call to control token usage.

* `--log-level <LEVEL>`
  One of `DEBUG`, `INFO`, `WARNING`, `ERROR`. Default `INFO`.

## 3) Instruction Prompt Hints

Message-level example:

```
You are a classifier. Return ONLY valid JSON per schema. Do not include prose.
Task: single-label classification per message.
Labels: provided via user JSON "categories".
Output schema:
{
  "items": [
    { "message_id": int, "primary_category": str, "scores": {<label>: float} }
  ]
}
Rules:
- Assign exactly one primary_category per message.
- Include a score for every category; they should roughly sum to 1.0.
- Focus on user's intent; if unclear, use "other".
- If input text is empty, use primary_category="other" and uniform scores.
```

Session-level example:

```
You are a classifier. Return ONLY valid JSON per schema.
Task: single-label classification per session.
Labels: provided via user JSON "categories".
Output schema:
{
  "session_id": str,
  "primary_category": str,
  "scores": {<label>: float},
  "rationale": str
}
Rules:
- Consider the entire thread; prioritize user messages.
- If intents change, choose the dominant or final resolved intent.
- Use "other" if unclear or out of scope.
```

## 4) Incremental Processing

* **Sessions**: The pipeline computes `processed_upto` as the max message timestamp for each session. If a session receives new messages later, it’s automatically reclassified.
* **Messages**: The pipeline **skips** messages that already have a row in `message_classification` unless you pass `--reclassify-existing-messages`.

## 5) Extensibility Notes

* **Roles**: Defaults to classifying only `user` messages. Include `assistant`/`tool` if you want a fuller map of the conversation.
* **Tool calls**: If you want to include `tool_calls` metadata in classification, join/enrich in `db.py` and adjust the text sent to the model.
* **Traceability**: The ETag of the instruction blob is stored as `instructions_version` in the results.
* **Reliability**: API calls use retries for rate limits and transient errors.


