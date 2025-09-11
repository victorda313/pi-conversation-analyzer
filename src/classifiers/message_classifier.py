"""Per-message classifier using Azure OpenAI.

Exports a function `classify_messages` that takes messages and returns a
list of classification dicts compatible with database upserts.

Output schema expected from the model:
{
  "items": [
    {
      "message_id": 123,
      "primary_category": "billing",
      "scores": {"billing": 0.9, "technical_support": 0.1}
    }, ...
  ]
}

The instruction prompt should describe: categories, single-label target,
JSON output format, and that inputs may contain system/assistant/tool roles
which should generally be ignored for customer-intent statistics unless
otherwise specified. By default we classify only user messages.
"""
from __future__ import annotations
from typing import Any, Dict, List, Set

from ..openai_client import classify_text

def _coerce_results(
    model_out: Dict[str, Any],
    categories: List[str],
    expected_ids: List[int],
) -> List[Dict[str, Any]]:
    """Validate and coerce model output to a complete, safe result set.

    - Keeps only items whose message_id is in expected_ids.
    - Ensures each expected_id is present exactly once.
    - Fills missing items with a default 'other' classification and uniform-ish scores.

    Args:
        model_out: Parsed dict from the model (expected to contain 'items': [...]).
        categories: Allowed labels.
        expected_ids: Message IDs we asked the model to classify.

    Returns:
        List[dict] with one entry per expected_id in the same order, each having:
        { "message_id": int, "primary_category": str, "scores": {label: float} }
    """
    items = model_out.get("items", [])
    by_id: Dict[int, Dict[str, Any]] = {}

    # Keep only valid IDs we asked for
    expected_set: Set[int] = set(int(i) for i in expected_ids)
    for it in items:
        try:
            mid = int(it.get("message_id"))
        except Exception:
            continue
        if mid in expected_set and mid not in by_id:
            # normalize structure
            primary = it.get("primary_category") or "other"
            scores = it.get("scores") or {}
            # Ensure all categories exist in scores; default to 0
            scores = {c: float(scores.get(c, 0.0)) for c in categories}
            # If all zeros, make it uniform-ish
            if sum(scores.values()) == 0.0:
                scores = {c: 1.0 / len(categories) for c in categories}
            by_id[mid] = {
                "message_id": mid,
                "primary_category": primary if primary in categories else "other",
                "scores": scores,
            }

    # Fill any missing with safe defaults
    default_scores = {c: 1.0 / len(categories) for c in categories}
    out: List[Dict[str, Any]] = []
    for mid in expected_ids:
        if mid in by_id:
            out.append(by_id[mid])
        else:
            out.append({
                "message_id": mid,
                "primary_category": "other",
                "scores": dict(default_scores),
            })

    return out


def build_user_payload(categories: list[str], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build user payload for message classification.

    Args:
        categories: Allowed category labels.
        items: List of {message_id, text} dicts.

    Returns:
        Dict ready to be JSON-serialized and sent as the user content.
    """
    expected_ids = [it["message_id"] for it in items]
    return {
        "task": "single-label classification per message",
        "categories": categories,
        "expected_count": len(items),
        "expected_ids": expected_ids,
        "schema": {
            "items": [
                {
                    "message_id": "int",
                    "primary_category": "str",
                    "scores": {c: "float in [0,1]" for c in categories},
                }
            ]
        },
        "items": items,
        "instructions": (
            "Return ONLY valid JSON with an 'items' array of length equal to expected_count. "
            "Each element MUST correspond to one message_id from expected_ids, exactly once. "
            "Assign exactly one primary_category to each message, and provide a probability-like "
            "score for every category that sums ~1. If the text is off-topic or unclear, use 'other'. "
            "Do not include any extra keys or commentary."
        ),
    }


def classify_messages(
    *,
    client,
    model: str,
    system_instructions: str,
    categories: list[str],
    batch: List[Dict[str, Any]],
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """Classify a batch of messages.

    Args:
        client: Azure OpenAI client.
        model: Deployment/model name.
        system_instructions: Instruction text from blob.
        categories: Allowed labels.
        batch: List of {message_id, text}.
        max_tokens: Max tokens for response.
        temperature: Sampling temperature.

    Returns:
        List of dicts with keys: message_id, primary_category, scores.
    """
    payload = build_user_payload(categories, batch)
    out = classify_text(
        client,
        model=model,
        system_instructions=system_instructions,
        user_payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    expected_ids = [it["message_id"] for it in batch]
    coerced = _coerce_results(out, categories, expected_ids)
    return coerced
