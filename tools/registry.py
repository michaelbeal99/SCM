"""Tool Registry — defines and manages available tools for the agentic loop.

Tools are routing targets of the Dispatcher (schema = tool-call-v1),
not a pipeline stage. They are invoked inside the agentic loop only.
"""

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class ToolResult:
    """Normalized result from a tool execution."""
    success: bool
    data: Any = None
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class ToolDef:
    """Definition of a registered tool."""
    name: str
    description: str
    handler: Callable[..., ToolResult]
    sandbox: bool = True
    timeout_ms: int = 5000

    def execute(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given arguments."""
        start = time.time()
        try:
            result = self.handler(**args)
            result.duration_ms = (time.time() - start) * 1000
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


class ToolRegistry:
    """Registry of available tools for the agentic loop."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self._tools: dict[str, ToolDef] = {}
        self._register_builtins()

    def register(self, tool: ToolDef) -> None:
        """Register a tool definition."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with args."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {tool_name}")
        return tool.execute(args)

    # ------------------------------------------------------------------
    # Built-in tools
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        """Register the standard set of built-in tools."""
        self.register(ToolDef(
            name="execute_python",
            description="Execute Python code in a sandboxed subprocess",
            handler=self._execute_python,
            sandbox=True,
            timeout_ms=5000,
        ))
        self.register(ToolDef(
            name="read_file",
            description="Read contents of a file",
            handler=self._read_file,
            sandbox=True,
            timeout_ms=3000,
        ))
        self.register(ToolDef(
            name="write_file",
            description="Write content to a file",
            handler=self._write_file,
            sandbox=True,
            timeout_ms=5000,
        ))
        self.register(ToolDef(
            name="list_directory",
            description="List files in a directory",
            handler=self._list_directory,
            sandbox=True,
            timeout_ms=3000,
        ))
        self.register(ToolDef(
            name="web_search",
            description="Search the web (stub — returns info message)",
            handler=self._web_search,
            sandbox=True,
            timeout_ms=10000,
        ))
        self.register(ToolDef(
            name="run_bash",
            description="Execute a restricted bash command",
            handler=self._run_bash,
            sandbox=True,
            timeout_ms=10000,
        ))

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _execute_python(self, code: str, **kwargs) -> ToolResult:
        """Execute Python code in a subprocess with timeout."""
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=str(self.project_root),
            )
            return ToolResult(
                success=result.returncode == 0,
                data={
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                    "returncode": result.returncode,
                },
                error=result.stderr.strip() if result.returncode != 0 else "",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Python execution timed out (5s)")

    def _read_file(self, path: str, **kwargs) -> ToolResult:
        """Read a file, restricted to project directory."""
        file_path = self._resolve_safe(path)
        if file_path is None:
            return ToolResult(success=False, error=f"Access denied: {path}")
        try:
            content = file_path.read_text()
            return ToolResult(success=True, data={"content": content, "path": str(file_path)})
        except FileNotFoundError:
            return ToolResult(success=False, error=f"File not found: {path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _write_file(self, path: str, content: str, **kwargs) -> ToolResult:
        """Write to a file, restricted to project directory."""
        file_path = self._resolve_safe(path)
        if file_path is None:
            return ToolResult(success=False, error=f"Access denied: {path}")
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return ToolResult(success=True, data={"path": str(file_path), "written": len(content)})
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _list_directory(self, path: str = ".", **kwargs) -> ToolResult:
        """List files in a directory, restricted to project root."""
        dir_path = self._resolve_safe(path)
        if dir_path is None:
            return ToolResult(success=False, error=f"Access denied: {path}")
        try:
            entries = []
            for entry in sorted(dir_path.iterdir()):
                entry_type = "dir" if entry.is_dir() else "file"
                entries.append({"name": entry.name, "type": entry_type})
            return ToolResult(success=True, data={"path": str(dir_path), "entries": entries})
        except FileNotFoundError:
            return ToolResult(success=False, error=f"Directory not found: {path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _web_search(self, query: str, **kwargs) -> ToolResult:
        """Stub: web search not implemented yet."""
        return ToolResult(
            success=True,
            data={"message": f"Web search stub: query='{query}'. "
                   "Full implementation deferred to Phase 4."}
        )

    # Whitelist of safe bash commands
    SAFE_COMMANDS = {"ls", "cat", "head", "tail", "wc", "echo", "pwd",
                     "date", "whoami", "uname", "df", "du", "find",
                     "grep", "sort", "uniq", "cut", "tr", "awk", "sed"}

    def _run_bash(self, command: str, **kwargs) -> ToolResult:
        """Execute a restricted bash command."""
        # Extract the base command (first word)
        base_cmd = command.strip().split()[0] if command.strip() else ""
        if base_cmd not in self.SAFE_COMMANDS:
            return ToolResult(
                success=False,
                error=f"Command not allowed: {base_cmd}. "
                      f"Allowed: {', '.join(sorted(self.SAFE_COMMANDS))}"
            )
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(self.project_root),
            )
            return ToolResult(
                success=result.returncode == 0,
                data={
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                    "returncode": result.returncode,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Command timed out (10s)")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _resolve_safe(self, path: str) -> Path | None:
        """Resolve a path, ensuring it stays within the project root."""
        try:
            resolved = (self.project_root / path).resolve()
            # Must be within project_root
            if str(resolved).startswith(str(self.project_root)):
                return resolved
            return None
        except (ValueError, OSError):
            return None
