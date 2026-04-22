"""Example: number-guess domain — action model fill.

Overlay pattern:
    The template ships `ActionCommand` with INSPECT/SUBMIT/NOOP.
    Domain adds one command: GUESS.

Diff against template `models.py`:
    + GUESS in ActionCommand enum
    + (optional) subclass Action to type-check `args["value"]`
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

# Import from template (don't duplicate base classes)
from models import Action as _BaseAction


class NumberGuessCommand(str, Enum):
    """Extended command set for number-guess domain."""
    INSPECT = "inspect"   # retained from template
    SUBMIT = "submit"     # retained
    NOOP = "noop"         # retained
    GUESS = "guess"       # NEW — args: {"value": int}


class NumberGuessAction(_BaseAction):
    """Typed action. The base `Action.args` is a free dict; we don't constrain
    at the schema level here — rule engine validates `args["value"]` instead.
    """

    # All fields inherited from _BaseAction (metadata, command, args, ...)
    # In the real template, `command` is `ActionCommand`; for this example,
    # callers use a NumberGuessCommand string which matches ActionCommand
    # values. Pydantic accepts the string regardless of enum class.
    pass
