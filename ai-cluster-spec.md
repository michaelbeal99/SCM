# AI Cluster — Full System Specification
**Project:** Personal AI Cluster  
**Owner:** Michael  
**Hardware:** Lenovo Gaming 3 — Ryzen 5 6600H / RTX 3050 4GB / Radeon 680M / 8GB DDR5 / 256GB NVMe  
**OS:** Debian 13 (Trixie) Headless  
**Status:** Active Development  
**Version:** 1.1

---

## 1. Project Philosophy

This system is built on a core thesis: **specialization beats scale**. Rather than running one large general-purpose model, a cluster of small, purpose-built models running in series can outperform a much larger generalist on specific tasks — while fitting on consumer-grade hardware.

Key principles:
- Every module stays in its domain. No module handles foreign content directly.
- Natural language crossings are explicit, typed, and routed.
- JSON is the universal wire format between all modules.
- Deterministic components replace model inference wherever possible.
- Compression is applied at every layer: vocabulary, quantization, architecture.

---

## 2. Hardware Profile

| Component | Spec | Role |
|---|---|---|
| CPU | AMD Ryzen 5 6600H (6c/12t) | Lightweight models, orchestration, tool sandbox |
| dGPU | NVIDIA RTX 3050 4GB GDDR6 | Primary inference, specialist models |
| iGPU | AMD Radeon 680M (RDNA2) | Secondary compute, router/dispatcher |
| RAM | 8GB DDR5 | CPU-side layers, orchestration state |
| Storage | 256GB NVMe (~3.5GB/s) | Model storage, fast swap |

### System RAM Constraint

The 8GB DDR5 ceiling is the second tightest budget after VRAM. Working allocation:

```
OS + kernel + drivers:           ~1.5GB
Ollama runtime + overhead:       ~1.0GB
Orchestration (LangGraph,        ~1.5GB
  SQLite, agent loop state):
Tool sandbox subprocess:         ~0.5GB
CPU draft model (Phase 5):       ~1.0GB
─────────────────────────────────────────
Reserved:                        ~5.5GB
Working headroom:                ~2.5GB
```

Speculative decoding with a 300M draft model on CPU is feasible but leaves limited headroom for parallel tool sandbox subprocesses. Heavy multi-process workflows must serialize or be deferred to dGPU.

### Memory Budget (VRAM) — Two States

The realistic VRAM picture differs significantly between dev-mode (stock Ollama models as placeholders) and post-fine-tuning (compressed CDV-trained models). Both must be tracked.

**Dev-mode (Phases 2–3):**
```
Dispatcher placeholder (Qwen2.5 1.5B, INT4):    ~1.0GB
Active Specialist (Qwen2.5-Coder 1.5B, INT4):   ~0.9GB
KV Cache + Runtime:                             ~0.5GB
────────────────────────────────────────────────────────
Peak usage:                                     ~2.4GB
Headroom remaining:                             ~1.6GB
```

**Post fine-tuning target (Phase 4+):**
```
Dispatcher (RWKV 430M, INT4):                   ~0.25GB
Active Specialist (CDV-compressed, INT4):       ~0.35GB
KV Cache + Runtime:                             ~0.50GB
────────────────────────────────────────────────────────
Peak usage:                                     ~1.10GB
Headroom remaining:                             ~2.90GB
```

The headroom unlocked by fine-tuning is what makes the ensemble strategy (§5) viable.

**Models explicitly excluded by the 4GB budget:** Phi-3 Mini (3.8B, ~2GB INT4) and sqlcoder-7b-2 (~4GB INT4). Both were considered and ruled out — they cannot coexist with a resident dispatcher.

---

## 3. System Architecture

```
USER INPUT (English)
        ↓
┌───────────────────────────────────┐
│          DISPATCHER               │
│  Mode 1: English → JSON Contract  │
│  Mode 2: NL Requests → Fragments  │
│  Mode 3: Goal → Plan Steps        │
│  Model: RWKV 430M (target)        │
│         Qwen2.5 1.5B (dev)        │
│  Always resident in VRAM          │
└──────────────┬────────────────────┘
               ↓ Typed JSON Contract
┌───────────────────────────────────┐
│    SPECIALIST (active only)       │
│  Receives: CDV JSON contract      │
│  Produces: Output + <NL> tokens   │
│  Swaps via Ollama automatically   │
└──────────────┬────────────────────┘
               ↓
┌───────────────────────────────────┐
│    PLACEHOLDER SCANNER            │
│    [Deterministic — no model]     │
│  Scans for <NL> tokens            │
│  Batches NL requests              │
│  Injects original intent context  │
└──────────────┬────────────────────┘
               ↓ Batched NL Request
┌───────────────────────────────────┐
│    DISPATCHER (Mode 2)            │
│  Same model, generation prompt    │
│  Already warm in VRAM             │
│  Returns {placeholder_id: text}   │
└──────────────┬────────────────────┘
               ↓
┌───────────────────────────────────┐
│    ASSEMBLER                      │
│    [Deterministic — no model]     │
│  Substitutes placeholders         │
│  Validates final output           │
└──────────────┬────────────────────┘
               ↓
         FINAL OUTPUT
```

**Tool Module placement:** Tools are a *routing target* of the Dispatcher (schema = `tool-call-v1`), not a stage in this linear pipeline. The Tool Call Module is described in §4.5 and is invoked via the agentic loop in §4.6, which sits above this pipeline.

---

## 4. Module Specifications

### 4.1 Dispatcher Module

**Purpose:** Single NL-capable model serving three operational modes. One model, three system prompts, zero swap cost between modes.

**Mode 1 — Dispatch:**
- Input: Raw English user request (single step)
- Process: Intent parsing → schema selection → field extraction
- Output: Typed JSON contract using target specialist's CDV vocabulary
- Model: RWKV 430M (post fine-tuning target) / Qwen2.5 1.5B (dev placeholder)

**Mode 2 — NL Generation:**
- Input: Batched NL requests + original intent context
- Process: Generate English fragments for each placeholder
- Output: `{placeholder_id: generated_text}` map
- Same model, different system prompt

**Mode 3 — Plan Decomposition:**
- Input: User goal (English) + specialist registry
- Process: Decompose multi-step goal into ordered dispatchable steps
- Output: 
  ```json
  {
    "steps": [
      {"id": 0, "intent": "string", "depends_on": []},
      {"id": 1, "intent": "string", "depends_on": [0]}
    ]
  }
  ```
- Same model, third system prompt
- Invoked by the agentic loop (§4.6) for goals that exceed a single dispatch

**Routing Logic (Mode 1 schema selection):**
```
Step 1: Rule-based fast path (~70% of requests)
  "python" / ".py" / code keywords → python-specialist-v1
  "sql" / "query" / "database"     → sql-specialist-v1
  "calculate" / "solve" / "math"   → math-specialist-v1
  "run" / "execute" / "file"       → tool-call-v1

Step 2: Embedding similarity fallback
  Unknown intent → nearest CDV match → schema selection

Step 3: Full model inference (ambiguous cases only)
  Dispatcher model resolves complex or multi-domain requests
```

**Training Approach:**
- Base: RWKV 430M (target) — Qwen2.5 1.5B used unmodified during dev
- Method: QLoRA fine-tuning via Unsloth
- Data: Synthetic pairs (English request → JSON contract) for Mode 1; synthetic (placeholder context → NL fragment) for Mode 2; synthetic (goal → step list) for Mode 3
- Generated using Claude API via OpenRouter
- Format: `{"input": "...", "output": {valid JSON}, "mode": 1|2|3}`

---

### 4.2 Specialist Modules

All specialists share these properties:
- Input: Always a typed JSON contract (CDV vocabulary only)
- Output: Domain content with `<NL>` placeholders for English fragments
- Never process raw English directly
- Managed by Ollama (automatic load/unload)
- Fine-tuned on domain-specific data only

#### Python Specialist

**Schema:** `python-specialist-v1`
```json
{
  "schema": "python-specialist-v1",
  "task": "generate | debug | refactor | explain | optimize",
  "intent": "[CDV terms only]",
  "inputs": [{"name": "string", "type": "CDV_type"}],
  "outputs": [{"type": "CDV_type"}],
  "constraints": ["CDV_term", "CDV_term"],
  "python_version": "3.11",
  "context": "optional existing code"
}
```

**Base model:** Qwen2.5-Coder 1.5B (primary) / DeepSeek-Coder 1.3B (alternative)  
**Quantization:** INT4 via llama.cpp (~700-900MB)  
**CDV vocabulary:** ~555 tokens (Python keywords + built-ins + ~200 intent delta words)  
**Training data:** Python-only code corpus + synthetic CDV-input/code-output pairs  

#### SQL Specialist

**Schema:** `sql-specialist-v1`
```json
{
  "schema": "sql-specialist-v1",
  "task": "generate | optimize | explain",
  "intent": "[CDV terms only]",
  "tables": ["table_name"],
  "filters": {},
  "output_format": "SELECT | INSERT | UPDATE | DELETE",
  "dialect": "postgresql | mysql | sqlite"
}
```

**Base model:** Qwen2.5-Coder 1.5B fine-tuned on SQL corpus  
**Quantization:** INT4 (~700-900MB)  

Note: sqlcoder-7b-2 was considered but does not fit the 4GB VRAM budget alongside a resident dispatcher. Fine-tuned Qwen2.5-Coder 1.5B on a SQL-only corpus is the production path.

#### Math Specialist

**Schema:** `math-specialist-v1`
```json
{
  "schema": "math-specialist-v1",
  "task": "solve | prove | simplify | calculate",
  "domain": "algebra | calculus | statistics | linear_algebra",
  "expression": "LaTeX or symbolic notation",
  "output_format": "step_by_step | result_only | code"
}
```

**Base model:** Qwen2.5-Math 1.5B  
**Quantization:** INT4 (~700-900MB)  

---

### 4.3 Controlled Domain Vocabulary (CDV)

The CDV is the typed vocabulary contract between the Dispatcher and each specialist. It eliminates linguistic ambiguity at the specialist boundary.

**Structure per specialist:**
```
Layer 1 — Language Keywords (free, already known):
  Python: while, for, if, else, try, except, class, def, return... (~35)

Layer 2 — Ecosystem Terms (free, already known):
  Python: list, dict, str, int, sorted, filter, map, range... (~400-500)

Layer 3 — Intent Delta (what we build):
  Words expressing human intent not in Layers 1-2
  Target: ~150-300 words per specialist
  Derived mathematically from user request corpus
```

**CDV Discovery Method (scikit-learn):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# 1. Collect user request corpus
# 2. Remove Layer 1 + Layer 2 vocabulary
# 3. TF-IDF matrix over remaining words
# 4. NMF to find latent intent dimensions
# 5. Top words per dimension = CDV candidates
# 6. Lasso regression to enforce sparsity
# 7. Final CDV: ~200 highest-potency words
```

**Word potency criteria:**
- Coverage: appears across many distinct intents
- Discriminability: presence changes code output
- Composability: combines productively with other CDV words
- Low ambiguity: one meaning in this domain

---

### 4.4 Placeholder Scanner

**Type:** Deterministic — no model required  
**Purpose:** Detect `<NL>` tokens in specialist output, batch into NL requests

**Placeholder format:**
```
<NL req="docstring|comment|string|error_msg|varname|log_msg"
    ctx="human description of needed content"
    max="token_count"
    tone="technical|user_facing|terse"
    style="snake_case|title_case|sentence">
```

**Process:**
1. Regex scan specialist output for `<NL ...>` tokens
2. Replace each with stable ID: `__NL_0__`, `__NL_1__` etc.
3. Extract attributes into `NLRequest` objects
4. Inject original intent JSON as context
5. Pass batched request to Dispatcher Mode 2
6. Receive `{id: content}` map
7. Substitute IDs back into output template

---

### 4.5 Tool Call Module

**Purpose:** Agentic capability — allows the system to take real-world actions

**Tool Registry (initial):**
```python
@tool("execute_python")   # Run Python in sandbox
@tool("read_file")        # Read file contents
@tool("write_file")       # Write file contents
@tool("web_search")       # Search the web
@tool("list_directory")   # List files in path
@tool("run_bash")         # Execute bash command (restricted)
```

**Tool Call Schema:**
```json
{
  "schema": "tool-call-v1",
  "tool": "tool_name",
  "args": {},
  "timeout_ms": 5000,
  "sandbox": true
}
```

**Sandbox constraints:**
- Python execution: RestrictedPython, 5s timeout, no network
- File access: restricted to project directory
- Bash: whitelist of safe commands only

**Execution pipeline:**
```
Tool call detected
  → Permission filter (is this tool allowed?)
  → Resource limiter (CPU time, memory caps)
  → Isolated executor (subprocess/RestrictedPython)
  → Result formatter (normalize to JSON)
  → Return to agentic loop
```

---

### 4.6 Agentic Loop

The agentic loop uses the Dispatcher in all three modes. There is no separate planner module — Mode 3 of the Dispatcher fills that role.

```python
def agent_loop(user_goal: str):
    plan = dispatcher.decompose(user_goal)        # Mode 3
    
    for step in plan["steps"]:
        ir = dispatcher.to_ir(step["intent"])      # Mode 1
        domain = ir["schema"]
        
        if domain == "tool-call-v1":
            result = tool_registry.execute(ir)
        else:
            skeleton = specialists[domain].run(ir)
            placeholders = scanner.scan(skeleton)
            fragments = dispatcher.generate_nl(placeholders)  # Mode 2
            result = assembler.assemble(skeleton, fragments)
        
        if goal_checker.complete(result, user_goal):
            return result
        
        plan = dispatcher.replan(plan, result)     # Mode 3 again
```

---

## 5. Quantization Strategy

### BitNet 1.58b (Research Track)
- Weights constrained to {-1, 0, +1}
- Must train from scratch — cannot quantize existing models
- All matrix multiplications become additions
- Exceptional CPU efficiency via SIMD
- Target: Router/Dispatcher replacement once proven
- Timeline: Phase 5

### TurboQuant / INT4 (Production Track)
- Post-training quantization of existing pretrained models
- No retraining required
- Applied to all specialists from day one
- Tool: llama.cpp (GGUF Q4_K_M format)
- Quality loss: minimal for domain-specific tasks

### Ensemble Strategy (Phase 5)
```
BitNet specialist (CPU):       fast path, ~70% of requests
TurboQuant specialist (GPU):   quality path, edge cases
Confidence arbitrator:         routes based on output certainty
```

This is only viable post fine-tuning, once the VRAM headroom shown in §2 (post fine-tuning state) opens up.

---

## 6. Speculative Decoding

Once base system is stable:
- Draft model: ~300M parameter model on CPU
- Verify model: Active specialist on GPU
- Draft generates 4-5 token candidates
- Verify accepts/rejects in single forward pass
- Target: 3-4x throughput improvement
- Particularly effective for repetitive code patterns

CPU draft model RAM cost is included in the system RAM budget in §2.

---

## 7. Technology Stack

| Layer | Tool | Purpose |
|---|---|---|
| Inference | Ollama | Model serving, automatic VRAM management |
| Orchestration | LangGraph | Agentic state machine |
| Schemas | Pydantic v2 | JSON contract validation |
| Fine-tuning | Unsloth | QLoRA on local hardware |
| Quantization | llama.cpp | GGUF conversion and inference |
| CDV Analysis | scikit-learn | NMF/Lasso vocabulary discovery |
| Storage | SQLite + SQLModel | Logging, caching, memory |
| UI | Gradio (dev) / CLI | Testing interface |
| Agent | Hermes Agent | Build orchestration and automation |
| Environment | venv + pip | Python dependency isolation |

---

## 8. Project File Structure

```
llm-cluster/
├── ai-cluster-spec.md         ← This document (lives in project root)
├── hermes-sop.md              ← Hermes operating rules
├── .gitignore
├── requirements.txt
├── .venv/                     ← Python virtual environment (ignored)
├── core/
│   ├── dispatcher.py          ← Mode 1 + Mode 2 + Mode 3
│   ├── scanner.py             ← Deterministic <NL> scanner
│   ├── assembler.py           ← Deterministic output assembly
│   ├── goal_checker.py        ← Agentic loop completion
│   └── schemas/
│       ├── base.py            ← Pydantic base contract
│       ├── python_schema.py   ← Python specialist contract
│       ├── sql_schema.py      ← SQL specialist contract
│       ├── math_schema.py     ← Math specialist contract
│       ├── tool_schema.py     ← Tool call contract
│       └── nl_schema.py       ← NL request/response
├── specialists/
│   ├── base.py                ← Specialist base class
│   ├── python_specialist.py
│   ├── sql_specialist.py
│   └── math_specialist.py
├── tools/
│   ├── registry.py            ← Tool definitions + decorators
│   ├── sandbox.py             ← Execution isolation
│   └── builtin/
│       ├── execute_python.py
│       ├── file_ops.py
│       └── web_search.py
├── cdv/
│   ├── python_cdv.json        ← Python CDV vocabulary
│   ├── sql_cdv.json
│   ├── math_cdv.json
│   └── analyzer.py            ← NMF/Lasso CDV builder
├── training/
│   ├── generate_data.py       ← Synthetic training data gen
│   ├── finetune.py            ← Unsloth QLoRA pipeline
│   └── evaluate.py            ← Benchmark against baselines
├── models/
│   └── modelfiles/            ← Ollama Modelfiles
│       ├── dispatcher
│       ├── python-specialist
│       ├── sql-specialist
│       └── math-specialist
├── agent/
│   └── loop.py                ← LangGraph agentic loop
├── tests/
│   ├── test_dispatcher.py
│   ├── test_scanner.py
│   ├── test_assembler.py
│   ├── test_specialists.py
│   └── test_tools.py
├── config.py                  ← Central configuration
├── main.py                    ← Entry point
└── README.md
```

---

## 9. Development Phases

### Phase 1 — Foundation
- [x] Hardware confirmed (RTX 3050, Radeon 680M)
- [x] Debian 13 headless installed
- [x] NVIDIA driver + CUDA confirmed
- [ ] Ollama installed and GPU inference confirmed
- [ ] Hermes Agent installed and configured

Phase 1 must be fully complete before Hermes begins Phase 2. Hermes verifies all Phase 1 boxes as the first action of its first session (see SOP §13).

### Phase 2 — Core Pipeline
- [x] Project structure initialized (dirs + venv + .gitignore + requirements.txt)
- [x] Spec and SOP placed in project root
- [x] Pydantic schemas for all contracts
- [x] Dispatcher Mode 1 (English → JSON) working
- [x] Python specialist running via Ollama
- [x] Placeholder Scanner (deterministic)
- [x] Dispatcher Mode 2 (NL generation)
- [x] Assembler (deterministic)
- [x] End-to-end: English → Python code working

### Phase 3 — Specialists + Tools
- [x] SQL specialist
- [x] Math specialist
- [x] Tool registry + sandbox
- [x] Dispatcher Mode 3 (plan decomposition)
- [x] Basic agentic loop
- [x] Multi-specialist routing confirmed

### Phase 4 — CDV + Fine-tuning
- [ ] Python CDV corpus analysis (NMF/Lasso)
- [ ] Synthetic training data generation
- [ ] Dispatcher fine-tuned on CDV contracts (all 3 modes)
- [ ] Python specialist fine-tuned on CDV input
- [ ] Benchmark vs baseline Gemma/general models
- [ ] Confirm post-fine-tuning VRAM budget hit (§2)

### Phase 5 — Compression Research
- [ ] BitNet specialist prototype
- [ ] Ensemble strategy (BitNet + TurboQuant)
- [ ] Speculative decoding implementation
- [ ] Full system benchmark

---

## 10. Benchmark Targets

| Task | Baseline (Gemma 7B) | Target (This System) |
|---|---|---|
| HumanEval Python | ~35-45% | >65% |
| SQL correctness | ~60% | >80% |
| Response latency | ~8-15s | <3s |
| VRAM usage | ~4GB | <2GB active (dev) / <1.2GB (post-FT) |
| Specialists available | 1 | 5+ |
