# AI Cluster — Full System Specification
**Project:** Personal AI Cluster  
**Owner:** Michael  
**Hardware:** Lenovo Gaming 3 — Ryzen 5 6600H / RTX 3050 4GB / Radeon 680M / 8GB DDR5 / 256GB NVMe  
**OS:** Debian 13 (Trixie) Headless  
**Status:** Active Development  

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

### Memory Budget (VRAM)
```
Always resident:
  Dispatcher/NL Model (INT4):    ~215MB
  Active Specialist (INT4):      ~175-500MB
  KV Cache + Runtime:            ~500MB
  ─────────────────────────────────────
  Peak usage:                    ~1.2GB
  Headroom remaining:            ~2.8GB
```

---

## 3. System Architecture

```
USER INPUT (English)
        ↓
┌───────────────────────────────────┐
│          DISPATCHER               │
│  Mode 1: English → JSON Contract  │
│  Mode 2: NL Requests → Fragments  │
│  Model: RWKV 430M or Phi-3 Mini   │
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

---

## 4. Module Specifications

### 4.1 Dispatcher Module

**Purpose:** Single NL-capable model serving two operational modes.

**Mode 1 — Dispatch:**
- Input: Raw English user request
- Process: Intent parsing → schema selection → field extraction
- Output: Typed JSON contract using target specialist's CDV vocabulary
- Model: RWKV 430M or Phi-3 Mini (fine-tuned)

**Mode 2 — NL Generation:**
- Input: Batched NL requests + original intent context
- Process: Generate English fragments for each placeholder
- Output: `{placeholder_id: generated_text}` map
- Same model, different system prompt — zero swap cost

**Routing Logic (schema selection):**
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
- Base: RWKV 430M or Phi-3 Mini 3.8B
- Method: QLoRA fine-tuning via Unsloth
- Data: Synthetic pairs (English request → JSON contract)
- Generated using Claude API via OpenRouter
- Format: `{"input": "English text", "output": {valid JSON contract}}`

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

**Base model:** DeepSeek-Coder 1.3B or Qwen2.5-Coder 1.5B  
**Quantization:** INT4 via llama.cpp (~350-650MB)  
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

**Base model:** sqlcoder-7b-2 or Qwen2.5-Coder 1.5B fine-tuned  
**Quantization:** INT4  

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
**Quantization:** INT4  

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

```python
def agent_loop(user_goal: str):
    plan = planner.decompose(user_goal)    # RWKV small
    
    for step in plan:
        ir = dispatcher.to_ir(step)
        domain = ir["domain"]
        
        if domain == "tool_call":
            result = tool_registry.execute(ir)
        else:
            skeleton = specialists[domain].run(ir)
            placeholders = scanner.scan(skeleton)
            fragments = dispatcher.generate_nl(placeholders)
            result = assembler.assemble(skeleton, fragments)
        
        if goal_checker.complete(result, user_goal):
            return result
        
        plan = planner.replan(plan, result)  # adapt on failure
```

---

## 5. Quantization Strategy

### BitNet 1.58b (Research Track)
- Weights constrained to {-1, 0, +1}
- Must train from scratch — cannot quantize existing models
- All matrix multiplications become additions
- Exceptional CPU efficiency via SIMD
- Target: Router/Dispatcher replacement once proven
- Timeline: Phase 4+

### TurboQuant / INT4 (Production Track)
- Post-training quantization of existing pretrained models
- No retraining required
- Applied to all specialists from day one
- Tool: llama.cpp (GGUF Q4_K_M format)
- Quality loss: minimal for domain-specific tasks

### Ensemble Strategy (Phase 3+)
```
BitNet specialist (CPU):       fast path, ~70% of requests
TurboQuant specialist (GPU):   quality path, edge cases
Confidence arbitrator:         routes based on output certainty
```

---

## 6. Speculative Decoding

Once base system is stable:
- Draft model: ~300M parameter model on CPU
- Verify model: Active specialist on GPU
- Draft generates 4-5 token candidates
- Verify accepts/rejects in single forward pass
- Target: 3-4x throughput improvement
- Particularly effective for repetitive code patterns

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

---

## 8. Project File Structure

```
llm-cluster/
├── core/
│   ├── dispatcher.py          ← Mode 1 + Mode 2
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
- [x] Ollama installed and GPU inference confirmed
- [x] Hermes Agent installed and configured
- [x] Project structure initialized

### Phase 2 — Core Pipeline
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
- [x] Basic agentic loop
- [x] Multi-specialist routing confirmed

### Phase 4 — CDV Discovery
- [ ] Collect Python task corpus (10,000+ real examples)
- [ ] Strip Layer 1 + Layer 2 vocabulary from corpus
- [ ] TF-IDF matrix over remaining words
- [ ] NMF analysis → latent intent dimensions
- [ ] Lasso regression → minimum predictive word set
- [ ] Mutual information scoring → rank by potency
- [ ] Final CDV delta: target ~200 words
- [ ] Build custom SentencePiece tokenizer (CDV + Python vocab)
- [ ] Validate tokenizer: encode/decode roundtrip on test corpus
- [ ] Document CDV in cdv/python_cdv.json

### Phase 5 — Training Data Generation
- [ ] Design CDV contract → code output pair format
- [ ] Generate 50,000+ synthetic training pairs via OpenCode Go
- [ ] Quality filter: syntax check all code outputs
- [ ] Quality filter: CDV compliance check all inputs
- [ ] Split: 90% train / 5% validation / 5% test
- [ ] Store as jsonlines format
- [ ] Baseline benchmark: qwen2.5-coder:1.5b on test set
- [ ] Document baseline scores before any training

### Phase 6 — Scratch Training: Python Specialist v1
- [ ] Select architecture: GPT-2 style transformer
- [ ] Determine minimum viable parameter count (target 10-50M)
- [ ] Configure nanoGPT or litGPT for CDV tokenizer
- [ ] Design training loop with checkpointing
- [ ] First training run: smallest viable model (~10M params)
- [ ] Evaluate on held-out test set
- [ ] Compare against qwen2.5-coder:1.5b baseline
- [ ] Iterate on size until benchmark target met
- [ ] Export to GGUF for Ollama integration
- [ ] Replace qwen2.5-coder:1.5b in pipeline with trained model
- [ ] Full pipeline test with scratch-trained specialist

### Phase 7 — Scratch Training: Additional Specialists
- [ ] SQL specialist: build CDV, generate data, train
- [ ] Math specialist: build CDV, generate data, train
- [ ] Dispatcher: train routing model from scratch
- [ ] Benchmark full scratch-trained pipeline vs baseline
- [ ] Document results

### Phase 8 — Compression + Optimization
- [ ] INT4 quantization of all scratch-trained models
- [ ] BitNet prototype: retrain smallest specialist with ternary weights
- [ ] Ensemble strategy: BitNet (CPU) + INT4 (GPU)
- [ ] Speculative decoding implementation
- [ ] Full system benchmark vs Gemma 7B baseline

---

## 10. The Core Experiment

The central hypothesis of this project:

> A specialist model trained from scratch on a constrained CDV vocabulary and narrow domain corpus can outperform a general model 10-75x its size on specific tasks.

| Metric | Baseline (qwen2.5-coder 1.5B) | Target (CDV Specialist ~20-50M) |
|---|---|---|
| HumanEval Python (narrow subset) | ~45% | >65% |
| SQL correctness (narrow subset) | ~60% | >80% |
| Model size | 986 MB | <100 MB |
| VRAM during inference | ~1GB | <200 MB |
| Tokens per second | ~20 | >60 |
| Specialists in VRAM simultaneously | 2 | 10+ |

If the hypothesis holds, the result is publishable and the architecture is validated.
