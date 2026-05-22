"""Agentic Loop — orchestrates the full multi-step pipeline.

Uses the Dispatcher in all three modes:
  Mode 3: decompose goal → step plan
  Mode 1: dispatch each step → typed contract
  Mode 2: generate NL fragments for specialist output

Integrates specialists (Python, SQL, Math) and tools from the registry.
"""

import re
from dataclasses import dataclass, field

from core.dispatcher import Dispatcher
from core.scanner import PlaceholderScanner
from core.assembler import Assembler
from core.schemas import ToolCallContract


@dataclass
class LoopResult:
    """Result of an agentic loop execution."""
    success: bool
    output: str = ""
    steps_executed: int = 0
    plan: dict = field(default_factory=dict)
    error: str = ""


class AgenticLoop:
    """Orchestrates multi-step goal execution using all three Dispatcher modes."""

    def __init__(self, project_root: str = "."):
        self.dispatcher = Dispatcher()
        self.scanner = PlaceholderScanner()
        self.assembler = Assembler()

        # Lazy-init specialists and tools to avoid circular imports
        self._python_specialist = None
        self._sql_specialist = None
        self._math_specialist = None
        self._tool_registry = None
        self._project_root = project_root

    @property
    def python_specialist(self):
        if self._python_specialist is None:
            from specialists.python_specialist import PythonSpecialist
            self._python_specialist = PythonSpecialist()
        return self._python_specialist

    @property
    def sql_specialist(self):
        if self._sql_specialist is None:
            from specialists.sql_specialist import SQLSpecialist
            self._sql_specialist = SQLSpecialist()
        return self._sql_specialist

    @property
    def math_specialist(self):
        if self._math_specialist is None:
            from specialists.math_specialist import MathSpecialist
            self._math_specialist = MathSpecialist()
        return self._math_specialist

    @property
    def tool_registry(self):
        if self._tool_registry is None:
            from tools.registry import ToolRegistry
            self._tool_registry = ToolRegistry(project_root=self._project_root)
        return self._tool_registry

    def run(self, goal: str) -> LoopResult:
        """Execute a goal through the full agentic loop.

        Args:
            goal: English goal description (any complexity).

        Returns:
            LoopResult with output, steps executed, and plan.
        """
        # Mode 3: decompose goal into steps
        plan = self.dispatcher.decompose(goal)
        if not plan.get("steps"):
            return LoopResult(
                success=False,
                error="Could not decompose goal",
                plan=plan,
            )

        results: list[str] = []
        steps_executed = 0

        for step in plan["steps"]:
            intent = step.get("intent", "")
            if not intent:
                continue

            # Mode 1: dispatch intent to a typed contract
            dispatch = self.dispatcher.to_ir(intent)
            contract = dispatch.contract

            # Route based on schema
            if isinstance(contract, ToolCallContract):
                # Execute tool
                tool_result = self.tool_registry.execute(
                    contract.tool, contract.args
                )
                if tool_result.success:
                    results.append(str(tool_result.data))
                else:
                    results.append(f"Tool error: {tool_result.error}")
            else:
                # Route to specialist
                skeleton = self._run_specialist(contract)

                # Strip markdown fences
                skeleton = self._strip_fences(skeleton)

                # Scanner — find <NL> placeholders
                intent_ctx = getattr(contract, "intent", "")
                scan = self.scanner.scan(skeleton, intent_context=intent_ctx)

                # Mode 2: generate NL fragments if needed
                if scan.requests:
                    fragments = self.dispatcher.generate_nl(scan.requests)
                else:
                    fragments = {}

                # Assembler
                result = self.assembler.assemble(scan.template, fragments)
                results.append(result)

            steps_executed += 1

        return LoopResult(
            success=True,
            output="\n\n".join(results),
            steps_executed=steps_executed,
            plan=plan,
        )

    def _run_specialist(self, contract) -> str:
        """Route a contract to the correct specialist."""
        schema = contract.schema
        if schema == "python-specialist-v1":
            return self.python_specialist.run(contract)
        elif schema == "sql-specialist-v1":
            return self.sql_specialist.run(contract)
        elif schema == "math-specialist-v1":
            return self.math_specialist.run(contract)
        else:
            # Default fallback to Python
            return self.python_specialist.run(contract)

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences from output."""
        text = re.sub(r"^```(?:python|sql|py)?\s*\n?", "", text.strip())
        text = re.sub(r"\n?\s*```\s*$", "", text)
        return text.strip()
