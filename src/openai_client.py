"""Azure OpenAI client factory and classification call helpers.

Provides thin wrappers around the OpenAI SDK to perform JSON-mode
classification given instructions and inputs.

Functions:
    build_client(cfg: Config) -> AzureOpenAI
    classify_text(client, model, system_instructions, user_payload, *, max_tokens, temperature) -> dict
"""
from __future__ import annotations
import json
import logging
from typing import Any, Dict, Optional, List

from openai import AzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError

logger = logging.getLogger(__name__)


def build_client(cfg) -> AzureOpenAI:
    """Construct an AzureOpenAI client from config.

    Args:
        cfg: Config object with endpoint, api key, and api version.

    Returns:
        AzureOpenAI client instance.
    """
    return AzureOpenAI(
        azure_endpoint=cfg.client_endpoint,
        api_key=cfg.client_api_key,
        api_version=cfg.client_api_version,
    )


def _robust_json_parse(content: str) -> Dict[str, Any]:
    """Try to parse JSON; if it fails, attempt small, safe repairs.

    Repairs:
        - Trim whitespace.
        - Remove a dangling trailing comma before a closing '}' or ']'.
        - Balance braces/brackets by appending the missing closers if the structure
          looks obviously truncated (e.g., ends mid-array/object).

    Args:
        content: String returned by the model.

    Returns:
        Parsed dict.

    Raises:
        json.JSONDecodeError if parsing still fails after repairs.
    """
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    s = content.strip()

    # Remove trailing comma before final closing bracket/brace:  ... ,]}
    # Do a few passes to clean common patterns.
    for _ in range(3):
        s = s.replace(",]", "]").replace(",}", "}")

    # If it still fails, try balancing braces/brackets (very conservative).
    # Count opens/closes ignoring quotes (rough heuristic).
    opens_curly = s.count("{")
    closes_curly = s.count("}")
    opens_square = s.count("[")
    closes_square = s.count("]")

    if opens_square > closes_square:
        s = s + ("]" * (opens_square - closes_square))
    if opens_curly > closes_curly:
        s = s + ("}" * (opens_curly - closes_curly))

    # Final attempt:
    return json.loads(s)

# KEEP your existing _robust_json_parse; add this new helper:

def _repair_json_with_model(
    client: AzureOpenAI,
    *,
    model: str,
    system_instructions: str,
    raw_text: str,
    schema_hint: Dict[str, Any],
    expected_ids: Optional[List[int]] = None,
    expected_count: Optional[int] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Ask the model to convert a malformed JSON-like string into valid JSON
    that conforms to the given schema. Returns parsed dict or raises.

    Args:
        client: Azure OpenAI client.
        model: Deployment/model name for the repair call (can be the same).
        system_instructions: Base instructions (we'll prepend a repair directive).
        raw_text: The malformed JSON-ish content to fix.
        schema_hint: A small JSON object describing the expected shape.
        expected_ids: Optional list of exact message_ids to include.
        expected_count: Optional number of items expected.
        max_tokens: Output cap for the repair call.
        temperature: Sampling temperature.

    Returns:
        Dict parsed from the repaired JSON.

    Raises:
        json.JSONDecodeError or OpenAI errors if still invalid.
    """
    # Build a concise repair directive
    directive = {
        "task": "repair-json",
        "requirements": [
            "Return ONLY valid JSON, no prose.",
            "Conform to the provided schema exactly.",
            "If fields are missing, fill with safe defaults.",
            "Only include items for the expected_ids (if provided), each exactly once.",
            "Do not add extra keys.",
        ],
        "schema": schema_hint,
        "expected_ids": expected_ids or [],
        "expected_count": expected_count,
        "input_maybe_json": raw_text,
    }

    messages = [
        {"role": "system", "content": "You are a JSON repair tool. Only output valid JSON."},
        {"role": "user", "content": json.dumps(directive, ensure_ascii=False)},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    repaired = resp.choices[0].message.content
    return _robust_json_parse(repaired)


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=1, max=20),
    retry=retry_if_exception_type((RateLimitError, APIError)),
)
def classify_text(
    client: AzureOpenAI,
    *,
    model: str,
    system_instructions: str,
    user_payload: Dict[str, Any],
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Call the model in JSON mode to classify text.

    The prompt is split into a system message (instructions) and a single
    user message containing a compact JSON payload (categories, texts, etc.).

    Args:
        client: Azure OpenAI client.
        model: Deployment/model name.
        system_instructions: The instruction text fetched from blob.
        user_payload: JSON-serializable payload to include as the user message.
        max_tokens: Max output tokens.
        temperature: Sampling temperature.

    Returns:
        Parsed JSON dict from the model's response.
    """
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        return _robust_json_parse(content)

    except Exception as exc:
        # One-shot repair attempt using a compact schema hint (+ expected ids/count if present)
        logger.exception("Failed to parse model JSON; attempting repair. Error: %s\nRaw: %s", exc, content)

        schema_hint: Dict[str, Any] = user_payload.get("schema", {})
        expected_ids = user_payload.get("expected_ids")
        expected_count = user_payload.get("expected_count")

        try:
            repaired = _repair_json_with_model(
                client,
                model=model,  # you can set a smaller/cheaper repair model here if you prefer
                system_instructions="",
                raw_text=content,
                schema_hint=schema_hint,
                expected_ids=expected_ids,
                expected_count=expected_count,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return repaired
        except Exception as exc2:
            logger.exception("Failed to repair model JSON: %s\nOriginal Raw: %s", exc2, content)
            raise