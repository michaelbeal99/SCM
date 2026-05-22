#!/usr/bin/env python3
"""Fetch code examples for tokenizer training."""
from datasets import load_dataset
import json, os

# HumanEval
print("Loading HumanEval...")
he = load_dataset("openai/openai_humaneval", split="test")
print(f"HumanEval: {len(he)} examples, keys: {list(he[0].keys())}")

# MBPP
print("Loading MBPP...")
try:
    mbpp = load_dataset("google-research-datasets/mbpp", "full", split="train")
    print(f"MBPP full: {len(mbpp)} examples")
except Exception as e:
    print(f"full failed: {e}")
    try:
        mbpp = load_dataset("mbpp", "sanitized", split="train")
        print(f"MBPP sanitized: {len(mbpp)} examples")
    except Exception as e2:
        print(f"sanitized failed: {e2}")
        mbpp = load_dataset("mbpp", split="train")
        print(f"MBPP default: {len(mbpp)} examples")

print(f"MBPP keys: {list(mbpp[0].keys())}")

# Collect code
code_samples = []

for item in he:
    if "canonical_solution" in item and item["canonical_solution"]:
        code_samples.append(item["canonical_solution"])
    if "prompt" in item and item["prompt"]:
        code_samples.append(item["prompt"])

for item in mbpp:
    if "code" in item and item["code"]:
        code_samples.append(item["code"])
    if "prompt" in item and item["prompt"]:
        code_samples.append(item["prompt"])

print(f"Total code samples: {len(code_samples)}")

# Save
out_path = "data/corpus/python_code.jsonl"
with open(out_path, "w") as f:
    for code in code_samples:
        f.write(json.dumps({"code": code}) + "\n")

print(f"Saved to {out_path}")
print(f"Sample 1: {code_samples[0][:200]}")
print(f"Sample 2: {code_samples[1][:200]}")
