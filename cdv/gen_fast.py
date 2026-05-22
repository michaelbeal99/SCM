"""High-throughput synthetic task generator — keeps model warm between calls."""
import json, re, sys, time
from pathlib import Path
import ollama

PROJECT = Path("/home/michael/SCM")
REAL = PROJECT / "data/corpus/python_tasks_real.jsonl"
OUT = PROJECT / "data/corpus/python_tasks_synthetic.jsonl"
MODEL = "qwen2.5-coder:1.5b"
TARGET = 9500  # synthetic target

PROMPTS = [
    "Generate 50 Python task descriptions. One per line. Vary: data structures, sorting, filtering, mapping, reducing, searching, counting, sets, tuples, dictionaries, list comprehensions, generators.",
    "Generate 50 Python task descriptions. One per line. Vary: string parsing, regex, file I/O, CSV, JSON, error handling, type checking, input validation, logging, configuration parsing.",
    "Generate 50 Python task descriptions. One per line. Vary: OOP classes, inheritance, decorators, context managers, async/await, threading, type hints, dataclasses, properties, abstract classes.",
    "Generate 50 Python task descriptions. One per line. Vary: recursion, dynamic programming, graph algorithms, tree traversal, binary search, greedy, backtracking, memoization, bit manipulation.",
    "Generate 50 Python task descriptions. One per line. Vary: web requests, APIs, JSON, database SQL, data validation, serialization, caching, rate limiting, authentication, environment variables.",
    "Generate 50 Python task descriptions. One per line. Vary: math calculations, statistics, probability, datetime, random numbers, hashing, encryption, encoding, prime numbers, fibonacci.",
    "Generate 50 Python task descriptions. One per line. Vary: testing pytest, CLI argparse, subprocess, benchmarking, profiling, image manipulation, text NLP, plotting, spreadsheets, PDF generation.",
    "Generate 50 Python task descriptions. One per line. Vary: functional programming, lambda, map, filter, reduce, currying, partial application, immutable data, pattern matching, pipeline composition.",
    "Generate 50 Python task descriptions. One per line. Vary: networking sockets, HTTP servers, websockets, message queues, pub/sub, event loops, coroutines, task scheduling, rate limiting.",
    "Generate 50 Python task descriptions. One per line. Vary: machine learning preprocessing, feature extraction, normalization, train/test split, cross-validation, metrics, confusion matrix, data augmentation.",
]

# Load existing
seen = set()
if OUT.exists():
    with open(OUT) as f:
        for line in f:
            try:
                t = json.loads(line)
                seen.add(t["text"][:80].lower().strip())
            except: pass

print(f"Existing synthetic: {len(seen)}", flush=True)

# Keep model warm by pre-loading it
print("Pre-loading model...", flush=True)
ollama.generate(model=MODEL, prompt="Hello", options={"num_predict": 1})
print("Model warm", flush=True)

batch = 0
t0 = time.time()
while len(seen) < TARGET:
    prompt = PROMPTS[batch % len(PROMPTS)]
    
    try:
        resp = ollama.generate(
            model=MODEL,
            prompt=prompt,
            options={"temperature": 0.85, "num_predict": 2560},
        )
        raw = resp["response"].strip()
        
        added = 0
        with open(OUT, "a") as f:
            for line in raw.split("\n"):
                line = line.strip()
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'^[-•*]\s*', '', line)
                line = line.strip('"').strip("'")
                if len(line) < 25: continue
                if line.lower().startswith(("here", "let me", "sure", "i can", "certainly")): continue
                key = line[:80].lower().strip()
                if key not in seen:
                    seen.add(key)
                    f.write(json.dumps({"text": line, "source": "synthetic"}) + "\n")
                    added += 1
        
        batch += 1
        elapsed = time.time() - t0
        rate = len(seen) / (elapsed / 3600) if elapsed > 0 else 0
        print(f"  batch {batch}: +{added} = {len(seen)}/{TARGET} ({rate:.0f}/hr)", flush=True)
        
        if len(seen) >= TARGET:
            break
            
    except Exception as e:
        print(f"  batch {batch} error: {e}", flush=True)
        batch += 1

elapsed = time.time() - t0
print(f"\nDone: {len(seen)} synthetic tasks in {elapsed:.0f}s ({len(seen)/(elapsed/3600):.0f}/hr)", flush=True)
