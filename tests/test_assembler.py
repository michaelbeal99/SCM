"""Tests for Assembler (deterministic) and Dispatcher Mode 2 (NL generation)."""
import pytest
from core.assembler import Assembler
from core.dispatcher import Dispatcher
from core.schemas import NLRequest


@pytest.fixture
def assembler():
    return Assembler()


@pytest.fixture
def dispatcher():
    return Dispatcher()


class TestAssembler:
    def test_substitutes_single_placeholder(self, assembler):
        template = 'def foo():\n    """__NL_0__"""'
        fragments = {"__NL_0__": "Sort a list of items."}
        result = assembler.assemble(template, fragments)
        assert "Sort a list of items." in result
        assert "__NL_0__" not in result

    def test_substitutes_multiple_placeholders(self, assembler):
        template = '"""__NL_0__"""\n# __NL_1__\nx = __NL_2__'
        fragments = {
            "__NL_0__": "Module docstring",
            "__NL_1__": "Initialize x",
            "__NL_2__": "42",
        }
        result = assembler.assemble(template, fragments)
        assert "Module docstring" in result
        assert "Initialize x" in result
        assert "42" in result
        assert "__NL_" not in result

    def test_no_placeholders_passes_through(self, assembler):
        template = "def foo():\n    return 42"
        result = assembler.assemble(template, {})
        assert result == template

    def test_validate_detects_unfilled(self, assembler):
        assert not assembler.validate("def foo():\n    __NL_0__")
        assert assembler.validate("def foo():\n    return 42")

    def test_extra_fragments_ignored(self, assembler):
        template = '"""__NL_0__"""'
        fragments = {"__NL_0__": "doc", "__NL_99__": "unused"}
        result = assembler.assemble(template, fragments)
        assert "doc" in result

    def test_missing_fragment_leaves_placeholder(self, assembler):
        template = '"""__NL_0__"""\n"""__NL_1__"""'
        fragments = {"__NL_0__": "only first"}
        result = assembler.assemble(template, fragments)
        assert "only first" in result
        assert "__NL_1__" in result  # unfilled

    def test_realistic_assembler_output(self, assembler):
        template = (
            'def sort_by_date(items):\n'
            '    """__NL_0__"""\n'
            '    date_key = "__NL_1__"\n'
            '    return sorted(items, key=lambda x: x.get(date_key))\n'
        )
        fragments = {
            "__NL_0__": "Sort a list of dictionaries by their date field.",
            "__NL_1__": "date",
        }
        result = assembler.assemble(template, fragments)
        assert "Sort a list of dictionaries by their date field." in result
        assert '"date"' in result
        assert "__NL_" not in result
        # Output should still be valid Python
        assert "def sort_by_date" in result


class TestDispatcherMode2:
    def test_generate_nl_empty_requests(self, dispatcher):
        result = dispatcher.generate_nl([])
        assert result == {}

    def test_generate_nl_single_request(self, dispatcher):
        """Live Ollama test: generate a docstring fragment."""
        requests = [
            NLRequest(
                placeholder_id="__NL_0__",
                req="docstring",
                ctx="sorts a list of dicts by date key",
                max=40,
                tone="technical",
                intent_context="sort list dict date",
            )
        ]
        fragments = dispatcher.generate_nl(requests)
        assert "__NL_0__" in fragments
        assert len(fragments["__NL_0__"]) > 0

    def test_generate_nl_multiple_requests(self, dispatcher):
        """Live Ollama test: batch generate multiple fragments."""
        requests = [
            NLRequest(
                placeholder_id="__NL_0__",
                req="docstring",
                ctx="sorts items by date",
                max=30,
            ),
            NLRequest(
                placeholder_id="__NL_1__",
                req="varname",
                ctx="date field name for sorting",
                max=10,
                style="snake_case",
            ),
        ]
        fragments = dispatcher.generate_nl(requests)
        assert "__NL_0__" in fragments
        assert "__NL_1__" in fragments
        assert len(fragments["__NL_0__"]) > 0
        assert len(fragments["__NL_1__"]) > 0

    def test_parse_fragments_json_format(self, dispatcher):
        raw = '{"__NL_0__": "Sort a list of items.", "__NL_1__": "date_key"}'
        requests = [
            NLRequest(placeholder_id="__NL_0__", req="docstring", ctx="sort"),
            NLRequest(placeholder_id="__NL_1__", req="varname", ctx="date"),
        ]
        fragments = dispatcher._parse_fragments(raw, requests)
        assert fragments["__NL_0__"] == "Sort a list of items."
        assert fragments["__NL_1__"] == "date_key"

    def test_parse_fragments_line_format(self, dispatcher):
        raw = "__NL_0__: Sort a list of dicts by date\n__NL_1__: date_field"
        requests = [
            NLRequest(placeholder_id="__NL_0__", req="docstring", ctx="sort"),
            NLRequest(placeholder_id="__NL_1__", req="varname", ctx="date"),
        ]
        fragments = dispatcher._parse_fragments(raw, requests)
        assert "Sort" in fragments["__NL_0__"]
        assert fragments["__NL_1__"] == "date_field"

    def test_parse_fragments_fills_missing(self, dispatcher):
        raw = '{"__NL_0__": "only one"}'
        requests = [
            NLRequest(placeholder_id="__NL_0__", req="docstring", ctx="first"),
            NLRequest(placeholder_id="__NL_1__", req="comment", ctx="second fallback"),
        ]
        fragments = dispatcher._parse_fragments(raw, requests)
        assert fragments["__NL_0__"] == "only one"
        assert fragments["__NL_1__"] == "second fallback"  # fallback to ctx
