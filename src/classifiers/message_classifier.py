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
from typing import Any, Dict, List

from ..openai_client import classify_text


def build_user_payload(categories: list[str], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build user payload for message classification.

    Args:
        categories: Allowed category labels.
        items: List of {message_id, text} dicts.

    Returns:
        Dict ready to be JSON-serialized and sent as the user content.
    """
    return {
        "task": "single-label classification per message",
        "categories": categories,
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
            "Assign exactly one primary_category to each message, and provide a"
            " probability-like score for every category that sums ~1. Focus on"
            " the user's intent. If the text is off-topic or unclear, use 'other'."
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
    return out.get("items", [])
