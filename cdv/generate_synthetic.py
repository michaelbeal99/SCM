"""Generate synthetic Python task descriptions via Ollama.

Appends to data/corpus/python_tasks.jsonl until target count reached.
"""
import json
import re
import sys
from pathlib import Path

import ollama

PROJECT = Path("/home/michael/SCM")
CORPUS_DIR = PROJECT / "data" / "corpus"
OUTPUT = CORPUS_DIR / "python_tasks.jsonl"
REAL_FILE = CORPUS_DIR / "python_tasks_real.jsonl"

# Load real tasks first
tasks = []
if REAL_FILE.exists():
    with open(REAL_FILE) as f:
        for line in f:
            tasks.append(json.loads(line))

# Also load any existing synthetic
if OUTPUT.exists():
    with open(OUTPUT) as f:
        for line in f:
            try:
                t = json.loads(line)
                if t.get("source") != "humaneval" and t.get("source") != "mbpp":
                    tasks.append(t)
            except:
                pass

# Deduplicate
seen = set()
unique = []
for t in tasks:
    key = t["text"][:80].lower().strip()
    if key not in seen:
        seen.add(key)
        unique.append(t)

print(f"Starting with {len(unique)} unique tasks", flush=True)

TARGET = 10500
SEEN_TEXTS = {t["text"][:80].lower().strip() for t in unique}

# Diverse prompt templates for varied output
PROMPTS = [
    "Generate 40 diverse Python programming task descriptions. One per line, numbered 1-40. "
    "Vary complexity and vocabulary. Include tasks about: data structures (lists, dicts, sets, tuples), "
    "sorting, filtering, mapping, reducing, grouping, searching, counting, finding duplicates, merging, splitting.",

    "Generate 40 Python task descriptions. One per line, numbered 1-40. Include tasks about: "
    "string manipulation (parsing, formatting, regex), file I/O (read, write, CSV, JSON), "
    "error handling (try/except, custom exceptions), type checking, input validation.",

    "Generate 40 Python task descriptions. One per line, numbered 1-40. Include tasks about: "
    "classes and OOP (inheritance, polymorphism, abstract classes), decorators, generators, "
    "context managers, iterators, properties, static methods, class methods, data classes.",

    "Generate 40 Python task descriptions. One per line, numbered 1-40. Include tasks about: "
    "recursion, dynamic programming, memoization, graph algorithms (BFS, DFS, Dijkstra), "
    "tree traversal, binary search, sliding window, two pointers, greedy algorithms, backtracking.",

    "Generate 40 Python task descriptions. One per line, numbered 1-40. Include tasks about: "
    "web requests, API calls, JSON processing, database queries, SQL generation, "
    "data validation, serialization, configuration parsing, environment variables, logging.",

    "Generate 40 Python task descriptions. One per line, numbered 1-40. Include tasks about: "
    "math (prime numbers, fibonacci, factorials, statistics, probability, linear algebra), "
    "datetime operations, random number generation, hashing, encryption, encoding.",

    "Generate 40 Python task descriptions. One per line, numbered 1-40. Include tasks about: "
    "async/await, threading, multiprocessing, subprocess management, "
    "CLI tools (argparse, click), testing (pytest, unittest, mocking), benchmarking, profiling.",

    "Generate 40 Python task descriptions. One per line, numbered 1-40. Include tasks about: "
    "image processing, text processing, NLP tasks, data visualization, plotting, "
    "spreadsheet manipulation, PDF generation, email handling, network programming, sockets.",
]

batch = 0
while len(unique) < TARGET:
    prompt = PROMPTS[batch % len(PROMPTS)]
    
    try:
        resp = ollama.generate(
            model="qwen2.5-coder:1.5b",
            prompt=prompt,
            options={"temperature": 0.85, "num_predict": 2048},
        )
        raw = resp["response"].strip()
        
        added = 0
        for line in raw.split("\n"):
            line = line.strip()
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            line = re.sub(r'^[-•*]\s*', '', line)
            line = line.strip('"').strip("'")
            
            if len(line) < 25:
                continue
            if line.lower().startswith(("here", "let me", "sure", "i can", "certainly", "of course")):
                continue
            if "generate" in line.lower() and "task" in line.lower():
                continue
            
            key = line[:80].lower().strip()
            if key not in SEEN_TEXTS:
                SEEN_TEXTS.add(key)
                unique.append({"text": line, "source": "synthetic"})
                added += 1
        
        batch += 1
        print(f"  batch {batch}: +{added} = {len(unique)}/{TARGET}", flush=True)
        
        # Save incrementally every 5 batches
        if batch % 5 == 0:
            with open(OUTPUT, "w") as f:
                for t in unique:
                    f.write(json.dumps(t) + "\n")
                    
    except Exception as e:
        print(f"  batch {batch} error: {e}", flush=True)
        batch += 1

# Final save
with open(OUTPUT, "w") as f:
    for t in unique:
        f.write(json.dumps(t) + "\n")

synthetic = len([t for t in unique if t["source"] == "synthetic"])
real = len([t for t in unique if t["source"] in ("humaneval", "mbpp")])
print(f"\nDone: {real} real + {synthetic} synthetic = {len(unique)} total", flush=True)
print(f"Saved to {OUTPUT}", flush=True)
