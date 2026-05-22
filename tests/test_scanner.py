"""Tests for Placeholder Scanner — deterministic <NL> detection."""
import pytest
from core.scanner import PlaceholderScanner, NL_PATTERN
from core.schemas import NLRequest, ScanResult


@pytest.fixture
def scanner():
    return PlaceholderScanner()


class TestNLPattern:
    def test_matches_full_placeholder(self):
        text = '<NL req="docstring" ctx="sorts a list" max="50" tone="technical" style="sentence">'
        match = NL_PATTERN.search(text)
        assert match is not None
        assert match.group("req") == "docstring"
        assert match.group("ctx") == "sorts a list"
        assert match.group("max") == "50"
        assert match.group("tone") == "technical"
        assert match.group("style") == "sentence"

    def test_matches_minimal_placeholder(self):
        text = '<NL req="comment" ctx="explain this">'
        match = NL_PATTERN.search(text)
        assert match is not None
        assert match.group("req") == "comment"
        assert match.group("ctx") == "explain this"

    def test_does_not_match_plain_text(self):
        text = "def foo(): pass"
        match = NL_PATTERN.search(text)
        assert match is None


class TestScanner:
    def test_finds_single_placeholder(self, scanner):
        output = 'def foo():\n    <NL req="docstring" ctx="sorts list" max="50">'
        result = scanner.scan(output)
        assert len(result.requests) == 1
        assert result.requests[0].req == "docstring"
        assert result.requests[0].ctx == "sorts list"
        assert "__NL_0__" in result.template
        assert "<NL" not in result.template

    def test_finds_multiple_placeholders(self, scanner):
        output = (
            'def bar(x):\n'
            '    <NL req="docstring" ctx="does a thing">\n'
            '    # <NL req="comment" ctx="inline note">\n'
            '    <NL req="varname" ctx="result variable" style="snake_case">\n'
            '    return 42'
        )
        result = scanner.scan(output)
        assert len(result.requests) == 3
        assert "__NL_0__" in result.template
        assert "__NL_1__" in result.template
        assert "__NL_2__" in result.template
        assert "<NL" not in result.template

    def test_no_placeholders_returns_empty(self, scanner):
        output = "def foo():\n    return 42"
        result = scanner.scan(output)
        assert len(result.requests) == 0
        assert result.template == output

    def test_placeholder_ids_are_sequential(self, scanner):
        output = (
            '<NL req="docstring" ctx="a">\n'
            '<NL req="comment" ctx="b">\n'
            '<NL req="string" ctx="c">'
        )
        result = scanner.scan(output)
        assert len(result.requests) == 3
        assert result.requests[0].placeholder_id == "__NL_0__"
        assert result.requests[1].placeholder_id == "__NL_1__"
        assert result.requests[2].placeholder_id == "__NL_2__"

    def test_injected_intent_context(self, scanner):
        output = '<NL req="docstring" ctx="sort items">'
        result = scanner.scan(output, intent_context="sort list dict date")
        assert result.requests[0].intent_context == "sort list dict date"

    def test_empty_output(self, scanner):
        result = scanner.scan("")
        assert len(result.requests) == 0
        assert result.template == ""

    def test_invalid_req_defaults_to_comment(self, scanner):
        output = '<NL req="novel" ctx="write a story">'
        result = scanner.scan(output)
        assert len(result.requests) == 1
        assert result.requests[0].req == "comment"

    def test_realistic_specialist_output(self, scanner):
        """Test with output similar to what the Python specialist produces."""
        output = (
            'def sort_by_date(items):\n'
            '    <NL req="docstring" ctx="sorts list of dicts by date key" max="30" tone="technical">\n'
            '    return sorted(items, key=lambda x: x.get(<NL req="varname" ctx="date field name" max="10" tone="terse" style="snake_case">))\n'
        )
        result = scanner.scan(output)
        assert len(result.requests) == 2
        # Verify template is valid Python (placeholders don't break syntax)
        assert "def sort_by_date" in result.template
        assert "__NL_0__" in result.template
        assert "__NL_1__" in result.template

    def test_scan_result_is_serializable(self, scanner):
        output = '<NL req="docstring" ctx="test">'
        result = scanner.scan(output)
        json_str = result.model_dump_json()
        assert "__NL_0__" in json_str
        # Round-trip
        sr2 = ScanResult.model_validate_json(json_str)
        assert sr2.requests[0].ctx == "test"
