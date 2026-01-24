
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

  lut = {}
  with open(input_path, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    for start, end in zip(boundaries[:-1], boundaries[1:]): # can parallelize here.
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

  counts = {}
  loc = {}
  for word_bytes in lut.keys():
    for pair in zip(word_bytes, word_bytes[1:]):
      if pair not in counts.keys():
        counts[pair] = 0
        loc[pair] = []
      counts[pair] += lut[word_bytes]
      loc[pair].append(word_bytes)

  #################################

  # lut --> how many times word occurs in text
  # counts --> how many times token appears in lut (text)
  # loc --> maps token to words that use the token

  while len(vocab) < vocab_size:

    # The problem here is how we are breaking ties.
    # I think our keys are integers and should be bytes
    # With integer it is returning (b' c', b'om') before (b't', b'h') because b' c' --> over 256.
    (merge_pair, merge_count) = max(counts.items(), key=lambda x: (x[1], x[0]))
    new_index = len(vocab)
    vocab[new_index] = merge_pair[0] + merge_pair[1]
    merges.append( merge_pair )
    #print (len(merges), merges[-1], merge_count, merge_pair, merge_pair[0] + merge_pair[1])

    print ('------------')
    print (merge_pair, counts[merge_pair])
    print ('------------')

    # update lut, loc, counts
    affected = loc.pop(merge_pair)
    for old_word_bytes in affected:
      new_word_bytes = merge(old_word_bytes, merge_pair, merge_pair[0] + merge_pair[1])
      print (old_word_bytes, new_word_bytes)

      # so the problem seems to be that we are changing the word (after merging tokens)
      # but the loc map dosnt reflect it
      # I think we need to re-think the data structure

      # you want to use the loc to find all the words where the pair occurs
      # I guess using the pair is more specific than using just the token
      # then we need to remove it from lut (word --> count)
      # and remove it from loc. actually I dont think we need to remove it from loc.
      # and we also need to remove it from counts (token --> count)
      # but we need to re-process each word that contained the pair
      # which means subtracting from count

      # the thing is, that word is always going to be in the loc pair list even though its stale
      # so we should just check to make sure its in lut (to check that its not stale)

      if old_word_bytes in lut.keys():
        # for lut, change word encoding, count remains the same
        lut[tuple(new_word_bytes)] = lut[old_word_bytes]
        count = lut.pop(old_word_bytes)
        # subtract count from all old word pairs
        for pair in zip(old_word_bytes, old_word_bytes[1:]):
          counts[pair] -= count
        # update counts based on new words 
        for pair in zip(new_word_bytes, new_word_bytes[1:]):
          if pair not in counts.keys():
            counts[pair] = 0
            loc[pair] = []
          counts[pair] += lut[tuple(new_word_bytes)]
          loc[pair].append(tuple(new_word_bytes))

  #################################

  return vocab, merges


