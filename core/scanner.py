"""Placeholder Scanner — deterministic, no model required.

Scans specialist output for <NL ...> placeholders, replaces them
with stable substitution IDs, and extracts NL request objects for
the Dispatcher (Mode 2) to fulfill.
"""

import re
from typing import Optional

from .schemas import NLRequest, ScanResult

# Regex to match <NL req="..." ctx="..." max="..." tone="..." style="...">
NL_PATTERN = re.compile(
    r"<NL\s+"
    r'req="(?P<req>[^"]*)"\s+'
    r'ctx="(?P<ctx>[^"]*)"'
    r'(?:\s+max="(?P<max>\d+)")?'
    r'(?:\s+tone="(?P<tone>[^"]*)")?'
    r'(?:\s+style="(?P<style>[^"]*)")?'
    r"\s*>"
)


class PlaceholderScanner:
    """Deterministic scanner for <NL ...> placeholders in specialist output."""

    # Valid req types per the schema
    VALID_REQ_TYPES = {"docstring", "comment", "string", "error_msg", "varname", "log_msg"}

    def scan(
        self,
        output: str,
        intent_context: Optional[str] = None,
    ) -> ScanResult:
        """Scan specialist output for <NL> placeholders and extract requests.

        Args:
            output: Raw specialist output containing <NL ...> tokens.
            intent_context: Original user intent to inject as context.

        Returns:
            ScanResult with template (placeholders replaced by IDs) and
            extracted NL request list.
        """
        # Pass 1: find all matches in document order and create requests
        matches = list(NL_PATTERN.finditer(output))
        requests: list[NLRequest] = []

        for i, match in enumerate(matches):
            req_raw: str = match.group("req")
            ctx: str = match.group("ctx")
            max_tokens: int = int(match.group("max")) if match.group("max") else 50
            tone_raw: str = match.group("tone") or "technical"
            style_raw: str = match.group("style") or "sentence"

            # Validate req type
            if req_raw not in self.VALID_REQ_TYPES:
                req_raw = "comment"

            nl_request = NLRequest(
                placeholder_id=f"__NL_{i}__",
                req=req_raw,  # type: ignore[arg-type]  # validated above
                ctx=ctx,
                max=max_tokens,
                tone=tone_raw,  # type: ignore[arg-type]  # Pydantic validates
                style=style_raw,  # type: ignore[arg-type]  # Pydantic validates
                intent_context=intent_context or "",
            )
            requests.append(nl_request)

        # Pass 2: build template by replacing <NL ...> with IDs (right to left
        # to preserve string indices)
        template = output
        for i in range(len(matches) - 1, -1, -1):
            match = matches[i]
            placeholder_id = f"__NL_{i}__"
            template = template[:match.start()] + placeholder_id + template[match.end():]

        return ScanResult(template=template, requests=requests)
