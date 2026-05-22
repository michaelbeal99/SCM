"""Phase 4 Step 1: Collect Python task corpus.

Sources:
  1. HumanEval (164 tasks)
  2. MBPP (~1,000 tasks)
  3. Synthetic generation via Ollama to reach 10,000+

Output: data/corpus/python_tasks.jsonl
"""

import json
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = CORPUS_DIR / "python_tasks.jsonl"

all_tasks = []

# ====================================================================
# Source 1: HumanEval
# ====================================================================
print("=== Source 1: HumanEval ===")
try:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval")
    for row in ds["test"]:
        prompt = row["prompt"]
        # Extract just the task description (the part after >>>)
        if ">>>" in prompt:
            desc = prompt.split(">>>")[1].strip()
            # Take the first meaningful line
            desc = desc.split("\n")[0].strip()
        else:
            desc = prompt[:200].strip()
        if len(desc) > 10:
            all_tasks.append({"text": desc, "source": "humaneval"})
    print(f"  HumanEval: {len([t for t in all_tasks if t['source'] == 'humaneval'])} tasks")
except Exception as e:
    print(f"  HumanEval failed: {e}")

# ====================================================================
# Source 2: MBPP
# ====================================================================
print("\n=== Source 2: MBPP ===")
try:
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")
    for split in ["train", "test", "validation"]:
        if split in ds:
            for row in ds[split]:
                text = row["text"]
                # Extract description (before "Write a function" or "Your code")
                desc = text.split("Write a function")[0].strip()
                desc = desc.split("Your code")[0].strip()
                if len(desc) > 10:
                    all_tasks.append({"text": desc, "source": "mbpp"})
    print(f"  MBPP: {len([t for t in all_tasks if t['source'] == 'mbpp'])} tasks")
except Exception as e:
    print(f"  MBPP failed: {e}")

# ====================================================================
# Source 3: Synthetic via Ollama
# ====================================================================
print(f"\n=== Source 3: Synthetic Generation ===")
current = len(all_tasks)
print(f"  Real datasets: {current} tasks")

# Build a diverse set of prompts to get varied output
GENERATION_PROMPTS = [
    "Generate 40 diverse Python programming task descriptions. One per line, numbered. "
    "Vary complexity, domains, and vocabulary. Focus on: data structures, algorithms, sorting, "
    "filtering, string manipulation, file I/O, error handling, list comprehensions, dictionaries, sets.",

    "Generate 40 Python task descriptions about: web scraping, API calls, JSON processing, "
    "CSV handling, data analysis, regular expressions, date/time operations, math calculations. "
    "One per line, numbered. Use varied verbs like 'implement', 'create', 'build', 'design', 'calculate'.",

    "Generate 40 Python task descriptions about: classes and OOP, decorators, generators, "
    "context managers, async/await, threading, multiprocessing, testing with pytest, "
    "type hints, dataclasses. One per line, numbered. Vary vocabulary.",

    "Generate 40 Python task descriptions about: recursion, dynamic programming, "
    "graph algorithms, tree traversal, searching, pathfinding, memoization, "
    "bit manipulation, combinatorics, optimization problems. One per line, numbered.",

    "Generate 40 Python task descriptions about: database queries, SQL generation, "
    "ORM usage, data validation, serialization, configuration parsing, "
    "logging, caching, rate limiting, authentication. One per line, numbered.",

    "Generate 40 Python task descriptions about: image processing, text processing, "
    "natural language tasks, PDF generation, Excel manipulation, plotting, "
    "CLI tools, argument parsing, environment variables, subprocess management. "
    "One per line, numbered.",
]

import ollama

batch_num = 0
target_total = 10500

while len(all_tasks) < target_total:
    # Cycle through prompt templates
    prompt_template = GENERATION_PROMPTS[batch_num % len(GENERATION_PROMPTS)]
    
    print(f"  Batch {batch_num+1}: {len(all_tasks)}/{target_total}...", end=" ", flush=True)
    
    try:
        response = ollama.generate(
            model="qwen2.5-coder:1.5b",
            prompt=prompt_template,
            options={"temperature": 0.9, "num_predict": 2048},
        )
        raw = response["response"].strip()
        
        # Parse numbered lines
        for line in raw.split("\n"):
            line = line.strip()
            # Remove leading numbers/bullets
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            # Filter out boilerplate and short lines
            if len(line) > 20 and not line.lower().startswith(("here", "let me", "sure", "i can")):
                all_tasks.append({"text": line, "source": "synthetic"})
        
        added = len([t for t in all_tasks if t["source"] == "synthetic"])
        print(f"synthetic={added}")
    except Exception as e:
        print(f"error: {e}")
    
    batch_num += 1
    if batch_num >= 300:  # safety limit
        break

# ====================================================================
# Write output
# ====================================================================
print(f"\n=== Writing corpus ===")

# Deduplicate (simple: by first 100 chars)
seen = set()
unique_tasks = []
for t in all_tasks:
    key = t["text"][:100].lower().strip()
    if key not in seen:
        seen.add(key)
        unique_tasks.append(t)

with open(OUTPUT, "w") as f:
    for task in unique_tasks:
        f.write(json.dumps(task) + "\n")

# Count by source
from collections import Counter
counts = Counter(t["source"] for t in unique_tasks)

print(f"  Output: {OUTPUT}")
print(f"  Total unique tasks: {len(unique_tasks)}")
for source, count in sorted(counts.items()):
    print(f"    {source}: {count}")

# Show samples
print(f"\n  First 5 samples:")
for i, task in enumerate(unique_tasks[:5]):
    print(f"    [{task['source']}] {task['text'][:120]}")
