"""
OpenAI Harmony format parser and builder.

Provides utilities for working with the Harmony response format
used by GPT-OSS models. Includes:

- Reasoning channel parsing (analysis/final sections)
- Output entry building for Responses API
- Harmony model detection

Contributed by LibraxisAI - https://libraxis.ai
"""

from __future__ import annotations

import re
import uuid
from typing import Any

# Optional openai-harmony import
try:
    from openai_harmony import Conversation, Message, Role, TextContent

    HARMONY_AVAILABLE = True
except ImportError:
    Conversation = Message = Role = TextContent = None  # type: ignore[assignment]
    HARMONY_AVAILABLE = False


# Regex for parsing reasoning channel markers
CHANNEL_PATTERN = re.compile(r"^\s*(analysis|final)\s*:?\s*$", re.IGNORECASE)

# Keywords that indicate a Harmony-format model
HARMONY_KEYWORDS = ["gpt-oss", "harmony"]


def is_harmony_model(model_name: str) -> bool:
    """
    Check if model uses Harmony response format.

    Args:
        model_name: Model identifier

    Returns:
        True if model uses Harmony format
    """
    model_lower = model_name.lower()
    return any(kw in model_lower for kw in HARMONY_KEYWORDS)


def parse_reasoning_channels(reasoning: str | None) -> tuple[str | None, str | None]:
    """
    Split Harmony-style plain text into analysis/final sections.

    Harmony models emit reasoning in format:
        analysis:
        [analysis/thinking text]
        final:
        [final response text]

    Args:
        reasoning: Raw reasoning text from model

    Returns:
        Tuple of (analysis_text, final_text), either may be None
    """
    if not reasoning:
        return None, None

    current_channel: str | None = None
    buffers: dict[str, list[str]] = {"analysis": [], "final": []}

    for line in reasoning.splitlines():
        match = CHANNEL_PATTERN.match(line)
        if match:
            current_channel = match.group(1).lower()
            # Include any text after the marker on same line
            remainder = line[match.end() :].strip()
            if remainder:
                buffers[current_channel].append(remainder)
            continue

        if current_channel:
            buffers[current_channel].append(line)

    analysis_text = "\n".join(buffers["analysis"]).strip() or None
    final_text = "\n".join(buffers["final"]).strip() or None

    return analysis_text, final_text


def build_harmony_output_entries(
    *,
    final_text: str | None,
    reasoning_text: str | None,
) -> list[dict[str, Any]]:
    """
    Create OpenAI Responses output[] entries with reasoning + message.

    Args:
        final_text: Main response text
        reasoning_text: Reasoning/analysis text (optional)

    Returns:
        List of output item dictionaries
    """
    outputs: list[dict[str, Any]] = []

    if reasoning_text:
        outputs.append(
            {
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "type": "reasoning",
                "status": "completed",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": reasoning_text}],
            }
        )

    if final_text is not None:
        outputs.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": final_text}],
            }
        )

    return outputs


def apply_harmony_parsing(result: dict[str, Any], model: str) -> dict[str, Any]:
    """
    Apply Harmony format parsing to chat completion result.

    If model uses Harmony format, extracts reasoning channels
    and restructures the response accordingly.

    Args:
        result: Raw chat completion result
        model: Model name/identifier

    Returns:
        Modified result with parsed Harmony content
    """
    if not is_harmony_model(model):
        return result

    # Extract content and reasoning from result
    choices = result.get("choices", [])
    if not choices:
        return result

    choice = choices[0]
    message = choice.get("message", {})

    content = message.get("content", "")
    reasoning = message.get("reasoning")

    # Try to parse reasoning channels
    if reasoning:
        analysis, final = parse_reasoning_channels(reasoning)
        if final:
            # Use parsed final as main content
            message["content"] = final
        if analysis:
            message["reasoning"] = analysis
    elif content:
        # Check if content itself has channel markers
        analysis, final = parse_reasoning_channels(content)
        if analysis or final:
            message["content"] = final or ""
            if analysis:
                message["reasoning"] = analysis

    return result


def create_harmony_conversation(
    messages: list[dict[str, str]],
) -> "Conversation | None":
    """
    Create an openai_harmony Conversation from chat messages.

    Requires openai-harmony package to be installed.

    Args:
        messages: List of {"role": str, "content": str} dicts

    Returns:
        Conversation object or None if harmony not available
    """
    if not HARMONY_AVAILABLE:
        return None

    harmony_messages: list[Message] = []

    for msg in messages:
        role_str = msg.get("role", "user").lower()

        try:
            role = Role(role_str)
        except ValueError:
            role = Role.USER

        content = msg.get("content", "")
        harmony_messages.append(
            Message.from_role_and_content(role, TextContent(text=content))
        )

    return Conversation.from_messages(harmony_messages)


def render_harmony_prompt(conversation: "Conversation") -> str:
    """
    Render Harmony conversation to prompt string.

    Args:
        conversation: Harmony Conversation object

    Returns:
        Formatted prompt string
    """
    if not HARMONY_AVAILABLE or conversation is None:
        return ""

    return conversation.render()


__all__ = [
    "HARMONY_AVAILABLE",
    "is_harmony_model",
    "parse_reasoning_channels",
    "build_harmony_output_entries",
    "apply_harmony_parsing",
    "create_harmony_conversation",
    "render_harmony_prompt",
]
