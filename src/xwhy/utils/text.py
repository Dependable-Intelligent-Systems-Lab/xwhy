"""Text utility functions."""

from typing import Literal


def inject_text_at_position(
    original_text: str,
    position: Literal["start", "middle", "end"],
    default_index: int = 4,
    inject_text: str | None = None,
) -> str:
    """Inject text into a string at a specific position based on a selection.

    Use a provided 'inject_text' string if available; otherwise, select a
    word from a default list using 'default_index'.

    Args:
        original_text: The base prompt or text.
        position: The placement target ("start", "middle", or "end").
        default_index: Index to select from ['could', 'would', 'should',
            '###', 'please']. Defaults to 4 ("please").
        inject_text: A custom string to inject. If provided, it overrides
            the default_index selection.

    Returns:
        str: The new text with the injected word/phrase.

    Raises:
        ValueError: If the position is invalid or default_index is out of range.

    """
    default_options = ["could", "would", "should", "###", "please"]

    # Selection logic
    if inject_text is not None:
        word_to_add = inject_text
    else:
        if 0 <= default_index < len(default_options):
            word_to_add = default_options[default_index]
        else:
            max_idx = len(default_options) - 1
            raise ValueError(f"default_index must be between 0 and {max_idx}")

    words = original_text.split()

    if position == "start":
        words.insert(0, word_to_add)
    elif position == "middle":
        # Calculate middle based on word count
        mid = len(words) // 2
        words.insert(mid, word_to_add)
    elif position == "end":
        words.append(word_to_add)
    else:
        raise ValueError("Position must be 'start', 'middle', or 'end'.")

    return " ".join(words)
