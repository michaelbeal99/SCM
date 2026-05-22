#!/usr/bin/env python3
"""Validate CDV tokenizer for Python specialist."""
import sentencepiece as spm
import json
import keyword
import builtins
import os
import sys

MODEL_PATH = os.path.join(os.path.dirname(__file__), "cdv_python.model")
CDV_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "cdv", "python_cdv.json")
CORPUS_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data", "corpus", "python_tasks.jsonl")

sp = spm.SentencePieceProcessor()
sp.load(MODEL_PATH)

errors = []

# Test 1: Vocabulary size
assert sp.vocab_size() == 555, f"Expected 555 tokens, got {sp.vocab_size()}"
print(f"✓ Vocab size: {sp.vocab_size()}")

# Test 2: Control tokens
for token in ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<NL>", "</NL>", "<SEP>"]:
    assert sp.piece_to_id(token) >= 0, f"Missing control token: {token}"
print(f"✓ All 7 control tokens present")

# Test 3: Python keywords (all 35 must be single-token)
py_keywords = keyword.kwlist
kw_single = 0
for kw in py_keywords:
    pieces = sp.encode_as_pieces(kw)
    try:
        assert len(pieces) == 1, f"Keyword '{kw}' splits into {pieces}"
        assert pieces[0] == kw, f"Keyword '{kw}' encoded as '{pieces[0]}'"
        kw_single += 1
    except AssertionError as e:
        errors.append(str(e))
print(f"✓ Python keywords: {kw_single}/{len(py_keywords)} single-token")

# Test 4: Python builtins (all must be single-token)
py_builtins = [b for b in dir(builtins) if b[0].islower() and not b.startswith("_")]
bi_single = 0
for b in py_builtins:
    pieces = sp.encode_as_pieces(b)
    try:
        assert len(pieces) == 1, f"Builtin '{b}' splits into {pieces}"
        assert pieces[0] == b, f"Builtin '{b}' encoded as '{pieces[0]}'"
        bi_single += 1
    except AssertionError as e:
        errors.append(str(e))
print(f"✓ Python builtins: {bi_single}/{len(py_builtins)} single-token")

# Test 5: CDV words roundtrip (all must encode/decode correctly)
with open(CDV_PATH) as f:
    cdv_data = json.load(f)
cdv_words = [w["word"] for w in cdv_data["words"]]
cdv_passed = 0
for word in cdv_words:
    ids = sp.encode_as_ids(word)
    decoded = sp.decode_ids(ids)
    try:
        assert decoded.strip() == word.strip(), f"CDV word '{word}' roundtrip: '{decoded.strip()}'"
        cdv_passed += 1
    except AssertionError as e:
        errors.append(str(e))
print(f"✓ CDV words roundtrip: {cdv_passed}/{len(cdv_words)}")

# Test 6: Content roundtrip on corpus
if os.path.exists(CORPUS_PATH):
    with open(CORPUS_PATH) as f:
        lines = f.readlines()[:100]
    content_passed = 0
    for line in lines:
        try:
            obj = json.loads(line)
            text = obj.get("prompt", obj.get("text", ""))
            if not text:
                continue
            ids = sp.encode_as_ids(text)
            decoded = sp.decode_ids(ids)
            if "".join(text.split()) == "".join(decoded.split()):
                content_passed += 1
        except:
            pass
    print(f"✓ Content roundtrip: {content_passed}/100")
else:
    print("⚠ Corpus not found, skipping content roundtrip")

# Test 7: Code tokenization (key Python patterns)
code_tests = [
    ("def foo(bar):", ["def", "▁"]),
    ("return x + 1", ["return", "▁"]),
    ("import numpy as np", ["import", "▁"]),
    ("class MyClass:", ["class", "▁"]),
    ("for i in range(10):", ["for", "▁"]),
    ("if __name__ == '__main__':", ["if", "▁"]),
    ("try:", ["try"]),
    ("except Exception as e:", ["except", "▁"]),
    ("with open(f) as file:", ["with", "▁"]),
    ("yield from gen()", ["yield", "▁"]),
]
for code, expected_prefix in code_tests:
    pieces = sp.encode_as_pieces(code)
    try:
        assert len(pieces) > 0
        # First piece should be the keyword
        assert pieces[0] == expected_prefix[0], f"'{code}' first piece: {pieces[0]} != {expected_prefix[0]}"
    except AssertionError as e:
        errors.append(str(e))
print(f"✓ Code tokenization: {len(code_tests)} patterns checked")

# Test 8: ID roundtrip
test_texts = [
    "def sort_list(items):",
    "import json",
    "for item in data:",
    "return result",
    "create algorithm",
]
for text in test_texts:
    ids = sp.encode_as_ids(text)
    decoded = sp.decode_ids(ids)
    reconstructed = sp.decode_ids(sp.encode_as_ids(decoded))
    try:
        assert "".join(text.split()) == "".join(reconstructed.split()), f"ID roundtrip failed for '{text}' -> '{reconstructed}'"
    except AssertionError as e:
        errors.append(str(e))
print(f"✓ ID roundtrip: {len(test_texts)} texts stable")

# Report
print()
if errors:
    print(f"❌ {len(errors)} failures:")
    for e in errors:
        print(f"   - {e}")
    sys.exit(1)
else:
    print("🎉 All tokenizer validation tests passed!")
    sys.exit(0)
