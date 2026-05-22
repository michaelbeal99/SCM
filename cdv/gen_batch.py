"""Generate one batch of synthetic Python task descriptions.

Usage: python3 gen_batch.py <batch_num>
Appends to data/corpus/python_tasks_synthetic.jsonl
"""
import json
import re
import sys
from pathlib import Path

import ollama

PROJECT = Path("/home/michael/SCM")
OUTPUT = PROJECT / "data" / "corpus" / "python_tasks_synthetic.jsonl"

PROMPTS = [
    "Generate exactly 35 diverse Python programming task descriptions. One per line. "
    "Include: data structures, sorting, filtering, string manipulation, file I/O, "
    "error handling, list comprehensions, dictionaries, sets, tuples.",

    "Generate exactly 35 Python task descriptions. One per line. "
    "Include: classes, OOP, decorators, generators, context managers, "
    "async/await, type hints, dataclasses, properties, inheritance.",

    "Generate exactly 35 Python task descriptions. One per line. "
    "Include: recursion, dynamic programming, graph algorithms, binary search, "
    "tree traversal, greedy algorithms, backtracking, memoization.",

    "Generate exactly 35 Python task descriptions. One per line. "
    "Include: web requests, JSON, CSV, API calls, database queries, "
    "data validation, serialization, configuration, logging, caching.",

    "Generate exactly 35 Python task descriptions. One per line. "
    "Include: math, statistics, probability, datetime, random numbers, "
    "hashing, encryption, encoding, prime numbers, fibonacci, factorials.",

    "Generate exactly 35 Python task descriptions. One per line. "
    "Include: testing, CLI tools, subprocess, threading, multiprocessing, "
    "benchmarking, profiling, image processing, text processing, NLP.",
]

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
prompt = PROMPTS[batch % len(PROMPTS)]

print(f"Batch {batch}: generating...", flush=True)

try:
    resp = ollama.generate(
        model="qwen2.5-coder:1.5b",
        prompt=prompt,
        options={"temperature": 0.85, "num_predict": 2048},
    )
    raw = resp["response"].strip()
    
    added = 0
    with open(OUTPUT, "a") as f:
        for line in raw.split("\n"):
            line = line.strip()
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            line = line.strip('"').strip("'")
            
            if len(line) < 25:
                continue
            if line.lower().startswith(("here", "let me", "sure", "i can", "certainly")):
                continue
            
            f.write(json.dumps({"text": line, "source": "synthetic"}) + "\n")
            added += 1
    
    # Count total
    with open(OUTPUT) as f:
        total = sum(1 for _ in f)
    print(f"Batch {batch}: +{added} = {total} total synthetic", flush=True)

except Exception as e:
    print(f"Batch {batch} error: {e}", flush=True)
