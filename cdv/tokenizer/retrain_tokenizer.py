#!/usr/bin/env python3
"""Rebuild tokenizer training data with actual Python code."""
import json, sentencepiece as spm, os, keyword, builtins

# 1. Load CDV words
with open("/home/michael/SCM/cdv/python_cdv.json") as f:
    cdv_data = json.load(f)
cdv_words = [w["word"] for w in cdv_data["words"]]
print(f"CDV words: {len(cdv_words)}")

# 2. Python keywords + builtins
py_keywords = keyword.kwlist
py_builtins = [b for b in dir(builtins) if b[0].islower() and not b.startswith("_")]
print(f"Python keywords: {len(py_keywords)}, builtins: {len(py_builtins)}")

# 3. Build training lines
training_lines = []

# Task descriptions from corpus (for NL coverage)
with open("/home/michael/SCM/data/corpus/python_tasks.jsonl") as f:
    for line in f:
        try:
            obj = json.loads(line)
            text = obj.get("prompt", obj.get("text", ""))
            if text:
                training_lines.append(text.strip())
        except:
            pass
print(f"Task descriptions: {len(training_lines)}")

# Actual Python code (for keyword coverage)
with open("/home/michael/SCM/data/corpus/python_code.jsonl") as f:
    for line in f:
        try:
            obj = json.loads(line)
            code = obj.get("code", "")
            if code:
                training_lines.append(code.strip())
        except:
            pass
print(f"Code samples: {len(training_lines) - len(cdv_words) if cdv_words else len(training_lines)}")

# Common Python patterns to reinforce keyword tokenization
patterns = [
    "def function_name(args):",
    "return result",
    "import module",
    "from package import name",
    "class ClassName:",
    "for item in iterable:",
    "while condition:",
    "if condition:",
    "elif other_condition:",
    "else:",
    "try:",
    "except Exception as e:",
    "with open(path) as f:",
    "yield value",
    "async def handler():",
    "await coroutine()",
    "raise ValueError(message)",
    "assert condition",
    "lambda x: expression",
    "not in collection",
    "is None",
    "and condition",
    "or condition",
    "pass",
    "break",
    "continue",
    "global variable",
    "nonlocal variable",
    "del item",
    "print(output)",
]
for _ in range(5):  # Repeat for frequency
    training_lines.extend(patterns)

# CDV words in context
context_templates = [
    "Write Python code to {}.",
    "Implement {} algorithm.",
    "Create {} function.",
    "Debug {} issue.",
    "Optimize {} performance.",
]
for word in cdv_words:
    for tmpl in context_templates:
        training_lines.append(tmpl.format(word))

print(f"Total training lines: {len(training_lines)}")

# Write training file
train_path = "/home/michael/SCM/cdv/tokenizer/training_text.txt"
with open(train_path, "w") as f:
    for line in training_lines:
        line = line.strip()
        if line:
            f.write(line + "\n")

print(f"Training file: {train_path} ({os.path.getsize(train_path):,} bytes)")

# Retrain tokenizer
model_prefix = "/home/michael/SCM/cdv/tokenizer/cdv_python"
spm.SentencePieceTrainer.train(
    input=train_path,
    model_prefix=model_prefix,
    vocab_size=555,
    model_type="unigram",
    character_coverage=1.0,
    input_sentence_size=20000,
    shuffle_input_sentence=True,
    num_sub_iterations=3,
    max_sentencepiece_length=16,
    split_by_unicode_script=True,
    split_by_number=True,
    split_by_whitespace=True,
    add_dummy_prefix=False,
    remove_extra_whitespaces=False,
    pad_id=0, bos_id=1, eos_id=2, unk_id=3,
    user_defined_symbols=["<NL>", "</NL>", "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SEP>"],
)

print(f"\nRetrained! Vocab: {model_prefix}.model ({os.path.getsize(model_prefix + '.model'):,} bytes)")

# Validate
sp = spm.SentencePieceProcessor()
sp.load(f"{model_prefix}.model")
print(f"Vocab size: {sp.vocab_size()}")

# Check Python keyword tokenization
print("\nPython keyword test:")
for kw in ["def", "return", "import", "class", "for", "while", "if", "elif", "else",
           "try", "except", "with", "yield", "async", "await", "raise", "pass",
           "break", "continue", "lambda", "from", "as", "in", "not", "is", "and", "or",
           "True", "False", "None"]:
    pieces = sp.encode_as_pieces(kw)
    status = "✓" if len(pieces) == 1 else f"✗ ({' '.join(pieces)})"
    print(f"  {kw:12s} → {status}")

# CDV coverage
full_token = sum(1 for w in cdv_words if len(sp.encode_as_pieces(w)) == 1 and sp.encode_as_pieces(w)[0] == w)
print(f"\nCDV single-token: {full_token}/{len(cdv_words)} ({100*full_token/len(cdv_words):.1f}%)")

# Content roundtrip
with open("/home/michael/SCM/data/corpus/python_tasks.jsonl") as f:
    test_lines = f.readlines()[:200]

passed = 0
for line in test_lines:
    try:
        obj = json.loads(line)
        text = obj.get("prompt", obj.get("text", ""))
        if text:
            ids = sp.encode_as_ids(text)
            decoded = sp.decode_ids(ids)
            if "".join(text.split()) == "".join(decoded.split()):
                passed += 1
    except:
        pass
print(f"Content roundtrip: {passed} tests passed")

# Compression ratio
all_ids = []
all_chars = 0
for line in test_lines:
    try:
        obj = json.loads(line)
        text = obj.get("prompt", obj.get("text", ""))
        if text:
            ids = sp.encode_as_ids(text)
            all_ids.extend(ids)
            all_chars += len(text)
    except:
        pass
print(f"Compression: {all_chars/len(all_ids):.1f} chars/token")
print(f"Vocab used: {len(set(all_ids))}/{sp.vocab_size()}")
