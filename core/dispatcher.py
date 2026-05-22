"""Dispatcher — single NL model serving three modes.

Mode 1: English → JSON contract (dispatch)
Mode 2: NL requests → fragments (generation)
Mode 3: Goal → plan steps (decomposition)

dev model: qwen2.5:1.5b via Ollama
target model: RWKV 430M fine-tuned
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import ollama

from .schemas import (
    PythonSpecialistContract,
    SQLSpecialistContract,
    MathSpecialistContract,
    ToolCallContract,
)

# ---------------------------------------------------------------------------
# Routing tables — keyword → schema selection (fast path, ~70% of requests)
# ---------------------------------------------------------------------------

PYTHON_KEYWORDS = [
    "python", ".py", "code", "function", "class", "def ", "script",
    "import", "package", "module", "program", "debug", "refactor",
    "optimize", "list", "dict", "sort", "filter", "map", "lambda",
    "decorator", "generator", "async", "await", "type hint",
]

SQL_KEYWORDS = [
    "sql", "query", "database", "table", "select", "insert",
    "update", "delete", "join", "schema", "index", "postgres",
    "mysql", "sqlite", "column", "row", "transaction",
]

MATH_KEYWORDS = [
    "math", "calculate", "solve", "equation", "algebra",
    "calculus", "statistics", "linear algebra", "integral",
    "derivative", "matrix", "vector", "proof", "theorem",
    "probability", "geometry",
]

TOOL_KEYWORDS = [
    "run", "execute", "file", "read", "write", "search",
    "list", "directory", "bash", "command", "terminal",
]

# Task-level keywords for refining the Python contract
TASK_KEYWORDS = {
    "generate": ["write", "create", "generate", "build", "make", "implement", "code"],
    "debug": ["debug", "fix", "bug", "error", "broken", "issue", "traceback"],
    "refactor": ["refactor", "rewrite", "restructure", "clean up", "improve"],
    "explain": ["explain", "describe", "what does", "how does", "understand"],
    "optimize": ["optimize", "faster", "speed", "performance", "efficient", "slow"],
}


# ---------------------------------------------------------------------------
# Keyword scoring
# ---------------------------------------------------------------------------

def _keyword_score(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear in the text (case-insensitive)."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def _best_task(text: str) -> str:
    """Determine the most likely Python task from keywords."""
    scores = {task: _keyword_score(text, kws) for task, kws in TASK_KEYWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "generate"


# ---------------------------------------------------------------------------
# Mode 1 — dispatch: English → validated JSON contract
# ---------------------------------------------------------------------------

DISPATCH_PROMPT = """You are a router that converts English requests into typed JSON contracts.

Available schemas and their fields:

python-specialist-v1:
  task: "generate" | "debug" | "refactor" | "explain" | "optimize"
  intent: CDV terms only (no full sentences)
  inputs: [{{"name": str, "type": str}}]
  outputs: [{{"type": str}}]
  constraints: [str]
  python_version: "3.11"
  context: optional existing code

sql-specialist-v1:
  task: "generate" | "optimize" | "explain"
  intent: CDV terms only
  tables: [str]
  filters: {{}}
  output_format: "SELECT" | "INSERT" | "UPDATE" | "DELETE"
  dialect: "postgresql" | "mysql" | "sqlite"

math-specialist-v1:
  task: "solve" | "prove" | "simplify" | "calculate"
  domain: "algebra" | "calculus" | "statistics" | "linear_algebra"
  expression: str
  output_format: "step_by_step" | "result_only" | "code"

tool-call-v1:
  tool: str
  args: {{}}
  timeout_ms: 5000
  sandbox: true

Output ONLY valid JSON. No explanation, no markdown fences.

Request: {request}"""


@dataclass
class DispatchResult:
    """Result of Mode 1 dispatch — carries the contract plus metadata."""
    contract: PythonSpecialistContract | SQLSpecialistContract | MathSpecialistContract | ToolCallContract
    routing_method: str  # "rule" | "keyword" | "model"
    confidence: float


@dataclass
class Dispatcher:
    """Single model, three modes. Uses qwen2.5:1.5b via Ollama."""

    model: str = "qwen2.5:1.5b"
    ollama_host: str = "http://localhost:11434"

    # -- Mode 1: dispatch ---------------------------------------------------

    def to_ir(self, request: str) -> DispatchResult:
        """Convert an English request into a validated typed JSON contract.

        Three-step routing:
        1. Rule-based fast path (high-confidence keyword match)
        2. Keyword scoring fallback (weaker match)
        3. Full model inference (ambiguous)
        """
        # Step 1: Rule-based fast path
        kw_scores = {
            "python-specialist-v1": _keyword_score(request, PYTHON_KEYWORDS),
            "sql-specialist-v1": _keyword_score(request, SQL_KEYWORDS),
            "math-specialist-v1": _keyword_score(request, MATH_KEYWORDS),
            "tool-call-v1": _keyword_score(request, TOOL_KEYWORDS),
        }

        best_schema = max(kw_scores, key=lambda k: kw_scores[k])
        best_score = kw_scores[best_schema]

        if best_score >= 3:
            # High confidence rule-based routing
            contract = self._build_rule_contract(best_schema, request)
            return DispatchResult(
                contract=contract,
                routing_method="rule",
                confidence=min(best_score / 10.0, 0.95),
            )

        if best_score >= 1:
            # Moderate confidence — keyword fallback
            contract = self._build_rule_contract(best_schema, request)
            return DispatchResult(
                contract=contract,
                routing_method="keyword",
                confidence=best_score / 10.0,
            )

        # Step 3: Low confidence — fall through to model inference
        return self._model_dispatch(request)

    def _build_rule_contract(self, schema: str, request: str):
        """Build a contract using keyword extraction without model inference."""
        intent = self._extract_intent(request)

        if schema == "python-specialist-v1":
            return PythonSpecialistContract(
                task=_best_task(request),
                intent=intent,
            )
        elif schema == "sql-specialist-v1":
            return SQLSpecialistContract(
                task="generate",
                intent=intent,
            )
        elif schema == "math-specialist-v1":
            return MathSpecialistContract(
                task="solve",
                domain="algebra",  # default, model would refine
                expression=intent,
            )
        elif schema == "tool-call-v1":
            return ToolCallContract(
                tool=self._infer_tool(request),
            )
        else:
            # Default to Python as safest fallback
            return PythonSpecialistContract(
                task="generate",
                intent=intent,
            )

    def _extract_intent(self, request: str) -> str:
        """Extract a concise intent string from the request.

        Removes stopwords and keeps only domain-relevant terms.
        In Phase 4 this will use CDV vocabulary.
        """
        # Simple: strip common stopwords and keep meaningful tokens
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "or", "but", "not", "so", "if", "then", "than",
            "that", "this", "these", "those", "can", "could", "would",
            "should", "will", "may", "might", "do", "does", "did",
            "please", "need", "want", "like", "just", "really", "very",
        }
        words = re.findall(r"[a-z_]+", request.lower())
        meaningful = [w for w in words if w not in stopwords and len(w) > 1]
        return " ".join(meaningful[:10])  # cap at 10 tokens

    def _infer_tool(self, request: str) -> str:
        """Infer the most likely tool from the request keywords."""
        tool_map = {
            "execute_python": ["python", "run", "execute", "code", "script"],
            "read_file": ["read", "open", "view", "show", "cat"],
            "write_file": ["write", "save", "create file", "output"],
            "web_search": ["search", "google", "find", "look up", "web"],
            "list_directory": ["list", "ls", "dir", "files", "directory"],
            "run_bash": ["bash", "shell", "command", "terminal", "run"],
        }
        scores = {tool: _keyword_score(request, kws) for tool, kws in tool_map.items()}
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "execute_python"

    def _model_dispatch(self, request: str) -> DispatchResult:
        """Use the LLM to determine the correct schema and build the contract."""
        prompt = DISPATCH_PROMPT.format(request=request)
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 256},
            )
            raw = response["response"].strip()
            # Remove markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)
        except Exception:
            # Fallback: if model fails, default to Python
            return DispatchResult(
                contract=PythonSpecialistContract(
                    task="generate",
                    intent=self._extract_intent(request),
                ),
                routing_method="model",
                confidence=0.1,
            )

        schema = data.get("schema", "python-specialist-v1")
        try:
            if schema == "python-specialist-v1":
                contract = PythonSpecialistContract(**{k: v for k, v in data.items() if k != "schema"})
            elif schema == "sql-specialist-v1":
                contract = SQLSpecialistContract(**{k: v for k, v in data.items() if k != "schema"})
            elif schema == "math-specialist-v1":
                contract = MathSpecialistContract(**{k: v for k, v in data.items() if k != "schema"})
            elif schema == "tool-call-v1":
                contract = ToolCallContract(**{k: v for k, v in data.items() if k != "schema"})
            else:
                contract = PythonSpecialistContract(task="generate", intent="unknown")
        except Exception:
            contract = PythonSpecialistContract(
                task="generate",
                intent=self._extract_intent(request),
            )

        return DispatchResult(
            contract=contract,
            routing_method="model",
            confidence=0.5,
        )

    # -- Mode 2: NL generation ---------------------------------------------

    def generate_nl(self, requests: list) -> "dict[str, str]":
        """Generate English fragments for batched NL requests.

        Args:
            requests: List of NLRequest objects from the Scanner.

        Returns:
            Mapping of placeholder_id → generated English text.
        """
        from .schemas import NLRequest

        if not requests:
            return {}

        # Build a batched prompt with all requests
        items = []
        for r in requests:
            items.append(
                f"ID: {r.placeholder_id}\n"
                f"Type: {r.req}\n"
                f"Context: {r.ctx}\n"
                f"Tone: {r.tone}\n"
                f"Style: {r.style}\n"
                f"Max tokens: {r.max}\n"
                f"Intent: {r.intent_context or 'N/A'}\n"
            )

        prompt = MODE2_PROMPT.format(
            batch="\n---\n".join(items),
            count=len(requests),
        )

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 512},
            )
            raw = response["response"].strip()
            return self._parse_fragments(raw, requests)
        except Exception:
            # Fallback: use context as the fragment directly
            return {r.placeholder_id: r.ctx for r in requests}

    def _parse_fragments(
        self, raw: str, requests: list
    ) -> "dict[str, str]":
        """Parse the model's NL generation response into a fragment map."""
        fragments: dict[str, str] = {}

        # Try JSON parsing first
        raw_clean = re.sub(r"^```(?:json)?\s*", "", raw)
        raw_clean = re.sub(r"\s*```$", "", raw_clean)
        try:
            data = json.loads(raw_clean)
            if isinstance(data, dict):
                for k, v in data.items():
                    if k.startswith("__NL_"):
                        fragments[k] = str(v)
        except json.JSONDecodeError:
            pass

        # If JSON parsing didn't fill everything, try line-based fallback
        if not fragments:
            current_id = None
            current_text: list[str] = []
            for line in raw.split("\n"):
                line = line.strip()
                id_match = re.match(r"^(__NL_\d+__)\s*[:=-]\s*(.+)", line)
                if id_match:
                    if current_id:
                        fragments[current_id] = " ".join(current_text)
                    current_id = id_match.group(1)
                    current_text = [id_match.group(2)]
                elif current_id and line:
                    current_text.append(line)
            if current_id:
                fragments[current_id] = " ".join(current_text)

        # Always fill any missing IDs with context fallback
        for r in requests:
            if r.placeholder_id not in fragments:
                fragments[r.placeholder_id] = r.ctx

        return fragments


MODE2_PROMPT = """You fill in natural-language placeholders in generated code.

For each placeholder ID below, generate the appropriate English text.
Output as a JSON object mapping ID to text. No markdown, no explanation.

{batch}

Generate exactly {count} fragments as JSON: {{"__NL_0__": "...", "__NL_1__": "..."}}"""
