"""Pipeline runner — ties together all Phase 2 components.

Orchestrates the full flow:
  English input → Dispatcher (Mode 1) → Specialist → Scanner
  → Dispatcher (Mode 2) → Assembler → Final output
"""

from .dispatcher import Dispatcher
from .scanner import PlaceholderScanner
from .assembler import Assembler
from .schemas import PythonSpecialistContract


class Pipeline:
    """End-to-end pipeline for English → code generation."""

    def __init__(self):
        self.dispatcher = Dispatcher()
        self.scanner = PlaceholderScanner()
        self.assembler = Assembler()

        # Lazy-import specialist to avoid circular deps at module load
        self._specialist = None

    @property
    def specialist(self):
        if self._specialist is None:
            from specialists.python_specialist import PythonSpecialist
            self._specialist = PythonSpecialist()
        return self._specialist

    def run(self, request: str) -> str:
        """Execute the full pipeline on an English request.

        Args:
            request: Natural language request (e.g., "write a Python
                     function to sort a list of dicts by date")

        Returns:
            Final assembled output with all <NL> placeholders filled.
        """
        # Step 1: Dispatch — route to the correct specialist
        dispatch = self.dispatcher.to_ir(request)
        contract = dispatch.contract

        # Step 2: Specialist — generate domain code with <NL> placeholders
        skeleton = self.specialist.run(contract)

        # Strip markdown fences the model sometimes wraps output in
        skeleton = self._strip_fences(skeleton)

        # Step 3: Scanner — detect <NL> tokens, create NL requests
        scan = self.scanner.scan(skeleton, intent_context=contract.intent)

        # Step 4: If there are placeholders, generate NL fragments
        if scan.requests:
            fragments = self.dispatcher.generate_nl(scan.requests)
        else:
            fragments = {}

        # Step 5: Assembler — substitute fragments into template
        result = self.assembler.assemble(scan.template, fragments)

        return self._strip_fences(result)

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Remove markdown code fences from model output."""
        import re
        text = re.sub(r"^```(?:python|py)?\s*\n?", "", text.strip())
        text = re.sub(r"\n?\s*```\s*$", "", text)
        return text.strip()
