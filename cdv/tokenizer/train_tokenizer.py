#!/usr/bin/env python3
"""Train CDV SentencePiece tokenizer for Python specialist."""
import sentencepiece as spm
import os, sys

train_path = "/home/michael/SCM/cdv/tokenizer/training_text.txt"
model_prefix = "/home/michael/SCM/cdv/tokenizer/cdv_python"

print(f"Training file: {train_path}")
print(f"File size: {os.path.getsize(train_path):,} bytes")

# Train SentencePiece unigram model — vocab_size=555 per spec
spm.SentencePieceTrainer.train(
    input=train_path,
    model_prefix=model_prefix,
    vocab_size=555,
    model_type="unigram",
    character_coverage=1.0,
    input_sentence_size=10000,
    shuffle_input_sentence=True,
    num_sub_iterations=2,
    max_sentencepiece_length=16,
    split_by_unicode_script=True,
    split_by_number=True,
    split_by_whitespace=True,
    add_dummy_prefix=False,
    remove_extra_whitespaces=False,
    pad_id=0,
    bos_id=1,
    eos_id=2,
    unk_id=3,
    user_defined_symbols=["<NL>", "</NL>", "<PAD>", "<BOS>", "<EOS>", "<UNK>", "<SEP>"],
)

print(f"\nModel saved: {model_prefix}.model ({os.path.getsize(model_prefix + '.model'):,} bytes)")
print(f"Vocab saved: {model_prefix}.vocab ({os.path.getsize(model_prefix + '.vocab'):,} bytes)")

# Load and inspect
sp = spm.SentencePieceProcessor()
sp.load(f"{model_prefix}.model")
print(f"\nVocabulary size: {sp.vocab_size()}")
print(f"Piece size: {sp.get_piece_size()}")

# How many CDV words appear as full tokens vs. broken into subwords?
import json
with open("/home/michael/SCM/cdv/python_cdv.json") as f:
    cdv_data = json.load(f)

cdv_words = [w["word"] for w in cdv_data["words"]]
full_token = 0
subword_token = 0
for word in cdv_words:
    pieces = sp.encode_as_pieces(word)
    if len(pieces) == 1 and pieces[0] == word:
        full_token += 1
    else:
        subword_token += 1

print(f"\nCDV word coverage:")
print(f"  Full single-token: {full_token}/{len(cdv_words)} ({100*full_token/len(cdv_words):.1f}%)")
print(f"  Broken into subwords: {subword_token}/{len(cdv_words)} ({100*subword_token/len(cdv_words):.1f}%)")

# Show sample encoding
print("\nSample encodings:")
for word in cdv_words[:10]:
    pieces = sp.encode_as_pieces(word)
    ids = sp.encode_as_ids(word)
    print(f"  {word:25s} → {pieces!r}  ids={ids}")

print("\nTraining complete!")
