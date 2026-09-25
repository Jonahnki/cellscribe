"""Exception types raised by Cellscribe.

Every error that is the user's to fix (bad path, malformed directory, invalid
config) derives from :class:`CellscribeError` so the CLI can print a clean,
specific message instead of a traceback.
"""

from __future__ import annotations


class CellscribeError(Exception):
    """Base class for user-facing Cellscribe errors."""


class InputFormatError(CellscribeError):
    """The input path does not match the expected structure for its format."""


class ConfigError(CellscribeError):
    """The configuration file or options are invalid."""


class PipelineError(CellscribeError):
    """A pipeline stage could not complete on this dataset."""
