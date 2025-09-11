"""Session-level classifier using Azure OpenAI.

Given the ordered list of messages (usually role=user), produce a single
category for the entire session.

Expected model output:
{
  "session_id": "abc123",
  "primary_category": "billing",
  "scores": {"billing": 0.72, "technical_support": 0.18, ...},
  "rationale": "Optional short rationale"
}
"""
from __future__ import annotations
from typing import Any, Dict, List

from ..openai_client import classify_text


def build_user_payload(
    *, session_id: str, categories: list[str], messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build the user payload for session classification.

    Args:
        session_id: The session identifier.
        categories: Allowed labels.
        messages: List of {role, content, timestamp} in chronological order.

    Returns:
        Dict ready to be JSON-serialized.
    """
    return {
        "task": "single-label session classification",
        "session_id": session_id,
        "categories": categories,
        "schema": {
            "session_id": "str",
            "primary_category": "str",
            "scores": {c: "float in [0,1]" for c in categories},
            "rationale": "str (<= 2 sentences)",
        },
        "messages": messages,
        "instructions": (
            "Decide the category that best represents the customer's overall intent"
            " across this session. Favor the user's messages over assistant/tool content."
            " If mixed, choose the dominant or final resolved intent. Use 'other' when unclear."
        ),
    }


def classify_session(
    *,
    client,
    model: str,
    system_instructions: str,
    categories: list[str],
    session_id: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Classify a single session.

    Args:
        client: Azure OpenAI client.
        model: Deployment/model name.
        system_instructions: Instruction text from blob.
        categories: Allowed labels.
        session_id: ID of the session being classified.
        messages: Chronologically ordered messages.
        max_tokens: Max tokens for response.
        temperature: Sampling temperature.

    Returns:
        Dict with keys: session_id, primary_category, scores, rationale.
    """
    payload = build_user_payload(session_id=session_id, categories=categories, messages=messages)
    return classify_text(
        client,
        model=model,
        system_instructions=system_instructions,
        user_payload=payload,
        max_tokens=max_tokens,
        temperature=temperature,
    )
