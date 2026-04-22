"""Log filters for the hackathon [START]/[STEP]/[END] format.

The YAML config references `DefaultHackathonTagFilter` — it inserts a default
`hackathon_tag` field on records that don't provide one, so the formatter
doesn't crash on library logs.
"""

from __future__ import annotations

import logging


class DefaultHackathonTagFilter(logging.Filter):
    """Ensure every LogRecord has a `hackathon_tag` attribute."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "hackathon_tag"):
            record.hackathon_tag = "LOG"
        return True
