
import numpy as np
from cs336_basics.pretokenization_helper import find_chunk_boundaries
import regex as re
from collections import defaultdict

#####################

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

#####################

# okay so the problem with our implementation is we did not do pre-tokenization

#PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_REGEX = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", re.UNICODE)

def train_bpe(input_path, vocab_size, special_tokens):
  #print ()
  #print (input_path)
  #print (vocab_size)
  #print (special_tokens)

  #print ('Brian Crafton')
  #assert False

  f = open(input_path, 'r')
  text = f.read()
  f.close()

  counts = defaultdict(int)
  for match in PAT_REGEX.finditer(text):
    token = match.group(0)
    b = token.encode("utf-8")
    word_bytes = tuple(bytes([x]) for x in b)
    counts[word_bytes] += 1

  print (counts.keys())

  text = text.encode('utf-8')
  indices = list(map(int, text))
  #print (text)

  vocab = { x: bytes([x]) for x in range(256) }
  merges = []

  #################################

  while len(vocab) + len(special_tokens) < vocab_size:

    counts = {}
    for index1, index2 in zip(indices, indices[1:]):
      pair = (index1, index2)
      if pair not in counts.keys():
        counts[pair] = 0
      counts[pair] += 1

    pair = max(counts, key=counts.get)

    #print (counts)
    #print (counts[pair])

    keys = list(counts.keys())
    keys = [ (vocab[a], vocab[b]) for (a, b) in keys ]
    values = list(counts.values())
    order = np.argsort(values)[::-1]

    '''
    print ()
    for i in order[0:5]:
      print ( keys[i], values[i] )
    assert False
    '''

    index1, index2 = pair
    new_index = len(vocab)
    vocab[new_index] = vocab[index1] + vocab[index2]
    merges.append(( vocab[index1], vocab[index2] ))
    indices = merge(indices, pair, new_index)

  #################################

  return vocab, merges


