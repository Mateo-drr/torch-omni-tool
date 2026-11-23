"""
Griffe extension that lets Google style docstrings omit blank lines before sections
The project writes docstrings with no internal blank lines, but the griffe Google parser only
detects a section header (Args:, Returns:, ...) when a blank line precedes it.
This extension inserts that blank line at docs build time, the source is left untouched.
Registered in mkdocs.yml under the mkdocstrings python handler extensions option.
"""

import re
from typing import Any

import griffe

_SECTION_NAMES = (
    "Args",
    "Arguments",
    "Params",
    "Parameters",
    "Keyword Args",
    "Keyword Arguments",
    "Other Args",
    "Other Arguments",
    "Other Params",
    "Other Parameters",
    "Returns",
    "Return",
    "Yields",
    "Yield",
    "Receives",
    "Receive",
    "Raises",
    "Raise",
    "Exceptions",
    "Except",
    "Warns",
    "Warnings",
    "Warning",
    "Attributes",
    "Attrs",
    "Functions",
    "Methods",
    "Classes",
    "Modules",
    "Examples",
    "Example",
    "Notes",
    "Note",
    "Deprecated",
)
_SECTION_RE = re.compile(rf"^(?:{'|'.join(_SECTION_NAMES)}):\s*$")


def add_section_blank_lines(text: str) -> str:
    """
    Insert a blank line before every section header that directly follows text
    Args:
        text (str): Cleaned docstring text
    Returns:
        str: Docstring text with a blank line above each section header
    """
    lines: list[str] = []
    for line in text.splitlines():
        if _SECTION_RE.match(line) and lines and lines[-1].strip():
            lines.append("")
        lines.append(line)
    return "\n".join(lines)


class CompactSections(griffe.Extension):
    """
    Make compact Google docstrings parseable by griffe
    """

    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        """
        Rewrite the docstring of every loaded object before it gets parsed
        Args:
            obj (griffe.Object): Module, class, function or attribute being loaded
            kwargs (Any): Other hook arguments, unused
        """
        if obj.docstring is not None:
            obj.docstring.value = add_section_blank_lines(obj.docstring.value)
