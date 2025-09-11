from __future__ import annotations
from typing import List, Dict

def strip_first_user_message_instructions(
    msgs_raw: List[Dict], split_criteria: str
) -> List[Dict]:
    """Strip embedded instructions from the first user message in a session.

    This function looks at the first message with role == "user" and, if the
    `split_criteria` string is found in its content, removes everything up to
    and including that marker. The message content is then replaced by the
    remaining substring.

    Args:
        msgs_raw: List of message dicts, each with at least
            {"id", "role", "content", "timestamp"}.
        split_criteria: A string used to split the first user message.
            Everything before (and including) this marker will be stripped.

    Returns:
        A new list of messages with the first user message cleaned.
        If no user message or no split marker is found, returns msgs_raw unchanged.
    """
    cleaned = list(msgs_raw)  # shallow copy
    for m in cleaned:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            content = m["content"]
            if split_criteria in content:
                # Keep only the part after the split marker
                parts = content.split(split_criteria, 1)
                if len(parts) == 2:
                    m["content"] = parts[1].strip()
            break  # only process the first user message
    return cleaned
