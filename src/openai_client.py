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
from typing import Any, Dict

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
        return json.loads(content)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse model JSON: %s\nRaw: %s", exc, content)
        raise
