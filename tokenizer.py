"""Baseline tokenizer: raw UTF-8 bytes, vocab of 256. Simple, never fails on
unseen text — and treats a Devanagari character as 3 tokens. Think about
what that does to your model's context window and your token budget on the
Hindi part of the corpus.

You may replace this with anything you train ON THE PROVIDED CORPUS ONLY
(e.g., BPE), as long as:
  1. it can encode ARBITRARY UTF-8 text (byte-level fallback) and it is
     LOSSLESS: decode(encode(text)) == text, exactly. The scorer and the
     graders both verify this round-trip — a lossy tokenizer makes bpb
     meaningless and disqualifies the run.
  2. this file keeps exposing:  load() -> tokenizer object with
     .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size.
     train.py and evaluate.py call load() with NO arguments — keep any
     extra parameters optional.
  3. anything it needs is saved under your submission folder and loaded by
     load() with no internet. Grading runs with cwd = your folder; resolve
     saved files relative to __file__ to be safe.
"""
import json
import os

class CharByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.char_to_id = {}
        self.id_to_char = {}

    def train(self, text):
        chars = sorted(list(set(text)))
        idx = 256
        for c in chars:
            if len(c.encode('utf-8')) > 1:
                self.char_to_id[c] = idx
                self.id_to_char[idx] = c
                idx += 1
        self.vocab_size = idx

    def encode(self, text):
        ids = []
        for c in text:
            if c in self.char_to_id:
                ids.append(self.char_to_id[c])
            else:
                ids.extend(list(c.encode("utf-8")))
        return ids

    def decode(self, ids):
        b_array = bytearray()
        result = []
        for i in ids:
            if i >= 256:
                if b_array:
                    result.append(b_array.decode("utf-8", errors="replace"))
                    b_array = bytearray()
                result.append(self.id_to_char.get(i, ""))
            else:
                b_array.append(i)
        if b_array:
            result.append(b_array.decode("utf-8", errors="replace"))
        return "".join(result)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char_to_id": self.char_to_id}, f)

    def load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.char_to_id = data["char_to_id"]
            self.id_to_char = {int(v): k for k, v in self.char_to_id.items()}
            if self.char_to_id:
                self.vocab_size = 256 + len(self.char_to_id)
            else:
                self.vocab_size = 256

def load(path="tokenizer.json"):
    tok = CharByteTokenizer()
    if os.path.exists(path):
        tok.load_vocab(path)
    return tok