"""Parse raw model output to extract the predicted tool name."""

from __future__ import annotations


def extract_predicted_tool(text: str, available_tool_names: set[str]) -> str:
    """Recover the predicted tool name from raw model output.

    Strategy:
    1. Strip any ``<think>...</think>`` block (Qwen3 thinking mode).
    2. Try an exact match on the first output line.
    3. Fall back to a substring search over all available tool names.
    4. Check for the literal word "none".
    5. Return the first line as-is (will be counted as invalid).
    """
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[-1]
    text = text.strip()
    first_line = text.split("\n")[0].strip().rstrip(".,;")
    if first_line in available_tool_names or first_line == "none":
        return first_line
    for name in sorted(available_tool_names):
        if name in text:
            return name
    if "none" in text.lower():
        return "none"
    return first_line
