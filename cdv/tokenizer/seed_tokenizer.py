#!/usr/bin/env python3
"""Build CDV tokenizer with seed vocabulary (forced single-token)."""
import json, sentencepiece as spm, os, keyword, builtins

# 1. Known tokens (must be single-token)
with open("/home/michael/SCM/cdv/python_cdv.json") as f:
    cdv_data = json.load(f)
cdv_words = [w["word"] for w in cdv_data["words"]]

py_keywords = keyword.kwlist
py_builtins = [b for b in dir(builtins) if b[0].islower() and not b.startswith("_")]
control = ["<NL>", "</NL>", "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SEP>"]

all_known = sorted(set(cdv_words) | set(py_keywords) | set(py_builtins) | set(control))
print(f"Known tokens: {len(all_known)} (CDV: {len(cdv_words)}, kw: {len(py_keywords)}, bi: {len(py_builtins)}, ctrl: {len(control)})")

# 2. Training text: actual code + descriptions
training_lines = []

with open("/home/michael/SCM/data/corpus/python_code.jsonl") as f:
    for line in f:
        try:
            obj = json.loads(line)
            code = obj.get("code", "")
            if code:
                training_lines.append(code.strip())
        except:
            pass

with open("/home/michael/SCM/data/corpus/python_tasks.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 2000:
            break
        try:
            obj = json.loads(line)
            text = obj.get("prompt", obj.get("text", ""))
            if text:
                training_lines.append(text.strip())
        except:
            pass

print(f"Training lines: {len(training_lines)}")

# Write training file
train_path = "/home/michael/SCM/cdv/tokenizer/training_text.txt"
with open(train_path, "w") as f:
    for line in training_lines:
        if line.strip():
            f.write(line.strip() + "\n")

# 3. Train with seed vocabulary
# vocab_size = len(all_known) + room for subword units
vocab_size = 555
model_prefix = "/home/michael/SCM/cdv/tokenizer/cdv_python"

spm.SentencePieceTrainer.train(
    input=train_path,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type="unigram",
    character_coverage=1.0,
    input_sentence_size=10000,
    shuffle_input_sentence=True,
    num_sub_iterations=3,
    max_sentencepiece_length=16,
    split_by_unicode_script=True,
    split_by_number=True,
    split_by_whitespace=True,
    add_dummy_prefix=False,
    remove_extra_whitespaces=False,
    pad_id=0, bos_id=1, eos_id=2, unk_id=3,
    user_defined_symbols=all_known,  # Force these as single tokens
)

print(f"Trained: {model_prefix}.model ({os.path.getsize(model_prefix + '.model'):,} bytes)")

# 4. Validate
sp = spm.SentencePieceProcessor()
sp.load(f"{model_prefix}.model")
print(f"Vocab size: {sp.vocab_size()}")

# Python keyword test
print("\nPython keywords:")
perfect = 0
for kw in py_keywords:
    pieces = sp.encode_as_pieces(kw)
    if len(pieces) == 1 and pieces[0] == kw:
        perfect += 1
        print(f"  ✓ {kw}")
    else:
        print(f"  ✗ {kw:15s} → {' '.join(pieces)}")
print(f"  Perfect: {perfect}/{len(py_keywords)}")

# Builtins test
print("\nPython builtins (sample):")
bi_perfect = 0
for b in py_builtins[:20]:
    pieces = sp.encode_as_pieces(b)
    if len(pieces) == 1 and pieces[0] == b:
        bi_perfect += 1
        print(f"  ✓ {b}")
    else:
        print(f"  ✗ {b:15s} → {' '.join(pieces)}")
# Count all
bi_total = sum(1 for b in py_builtins if len(sp.encode_as_pieces(b)) == 1)
print(f"  All builtins perfect: {bi_total}/{len(py_builtins)}")

# CDV test
cdv_perfect = sum(1 for w in cdv_words if len(sp.encode_as_pieces(w)) == 1)
print(f"\nCDV single-token: {cdv_perfect}/{len(cdv_words)} ({100*cdv_perfect/len(cdv_words):.1f}%)")

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
print(f"Content roundtrip: {passed}/200")

# Code test
print("\nCode tokenization:")
samples = ["def foo():", "return x + 1", "import os", "class Bar:", "for i in range(10):"]
for s in samples:
    pieces = sp.encode_as_pieces(s)
    ids = sp.encode_as_ids(s)
    print(f"  {s:30s} → {len(ids):2d} tokens: {pieces}")

# Stats
all_ids = []
for line in test_lines:
    try:
        obj = json.loads(line)
        text = obj.get("prompt", obj.get("text", ""))
        if text:
            all_ids.extend(sp.encode_as_ids(text))
    except:
        pass
print(f"\nCompression: {sum(len(obj['prompt']) for obj in [json.loads(l) for l in test_lines if json.loads(l).get('prompt')]) / len(all_ids):.1f} chars/token")
print(f"Vocab utilization: {len(set(all_ids))}/{sp.vocab_size()}")
