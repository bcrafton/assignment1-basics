
import numpy as np
from cs336_basics.pretokenization_helper import find_chunk_boundaries
import regex as re
from collections import defaultdict

#####################

def merge(text: list[bytes], pair: tuple[bytes, bytes], new_token: bytes) -> list[bytes]:
    new_text = []
    i = 0
    while i < len(text):
        if i + 1 < len(text) and text[i] == pair[0] and text[i + 1] == pair[1]:
            new_text.append(new_token)
            i += 2
        else:
            new_text.append(text[i])
            i += 1
    return new_text

#####################

# okay so the problem with our implementation is we did not do pre-tokenization

PAT_REGEX = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", re.UNICODE)
def train_bpe(input_path, vocab_size, special_tokens):
  print ()
  #print (special_tokens)

  split_pattern = re.compile("|".join(re.escape(st) for st in special_tokens))
  num_processes = 4
  vocab = { x: bytes([x]) for x in range(256) }
  vocab[256] = b'<|endoftext|>'
  merges = []

  with open(input_path, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    lut = {}
    for start, end in zip(boundaries[:-1], boundaries[1:]):
      f.seek(start)
      chunk = f.read(end - start).decode("utf-8", errors="ignore")
      parts = split_pattern.split(chunk)

      for part in parts:
        for match in PAT_REGEX.finditer(part):
          word = match.group(0)
          word = word.encode("utf-8")
          word_bytes = tuple(bytes([x]) for x in word)
          if word_bytes not in lut.keys():
            lut[word_bytes] = 0
          lut[word_bytes] += 1

  #################################

  while len(vocab) < vocab_size:
    # print (len(vocab))

    counts = {}
    for word_bytes in lut.keys():
      for pair in zip(word_bytes, word_bytes[1:]):
        if pair not in counts.keys():
          counts[pair] = 0
        counts[pair] += lut[word_bytes]
        #print (pair[0])
        assert type(pair[0]) == bytes
        assert type(pair[1]) == bytes

    # The problem here is how we are breaking ties.
    # I think our keys are integers and should be bytes
    # With integer it is returning (b' c', b'om') before (b't', b'h') because b' c' --> over 256.
    (pair, merge_count) = max(counts.items(), key=lambda x: (x[1], x[0]))
    new_index = len(vocab)
    assert type(pair[0]) == bytes
    assert type(pair[1]) == bytes
    vocab[new_index] = pair[0] + pair[1]

    merges.append( pair )
    #print (len(merges), merges[-1], merge_count, pair, pair[0] + pair[1])

    new_lut = {}
    for (word_bytes, count) in lut.items():
      word_bytes = merge(word_bytes, pair, pair[0] + pair[1])
      new_lut[tuple(word_bytes)] = count
    lut = new_lut

  #################################

  return vocab, merges


