"""
Used for both ISOT + LIAR and backend inference
Strip HTMLs, remove URLs, normalise whitespace
"""

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")

def clean_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text)

    # Remove HTML tags
    s = _HTML_RE.sub(" ", s)

    # Remove URLs
    s = _URL_RE.sub(" ", s)

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s