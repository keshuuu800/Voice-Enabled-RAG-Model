"""
Text cleaner for document preprocessing.

Cleans raw extracted text before chunking without losing meaningful content.
Designed to work on multilingual (English/Hindi/Hinglish) text.
"""
import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# Lines that are purely decorative separators (----, ====, .....) with 3+ chars
_SEPARATOR_PATTERN = re.compile(r"^[\-=_\*\.~#]{3,}\s*$", re.MULTILINE)


class TextCleaner:
    """Cleans extracted document text while preserving structure and meaning."""

    def clean(self, text: str) -> str:
        """Apply all cleaning steps in order.

        Steps:
        1. Normalize line endings
        2. Remove null bytes and control characters (keep \\n, \\t)
        3. Unicode normalization (NFKC)
        4. Normalize excessive blank lines
        5. Expand tabs
        6. Strip trailing whitespace per line
        7. Remove pure-separator lines
        8. Collapse multiple spaces
        9. Final strip

        Does NOT: lowercase, remove punctuation, or destroy tables.
        """
        if not text:
            return ""

        # 1. Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 2. Remove null bytes and non-printable control chars (keep \n \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 3. Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # 4. Collapse 3+ consecutive blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 5. Expand tabs to spaces
        text = text.expandtabs(4)

        # 6. Strip trailing whitespace from each line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # 7. Remove pure separator lines (---, ===, ..., etc.)
        text = _SEPARATOR_PATTERN.sub("", text)

        # 8. Collapse multiple consecutive spaces to one
        #    (but NOT newlines — preserve paragraph structure)
        text = re.sub(r" {2,}", " ", text)

        # 9. Final strip
        text = text.strip()

        return text

    def is_meaningful(self, text: str) -> bool:
        """Return True if text has enough non-whitespace content to be worth indexing."""
        return len(text.strip().replace("\n", "").replace(" ", "")) > 20
