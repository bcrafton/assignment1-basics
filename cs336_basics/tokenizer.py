
import regex as re
from collections.abc import Iterable, Iterator

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

'''
def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices
'''

# the problem appears to be that we need to merge to integers.
# so i think we should covert the bytes object to integers like the original

# no so actually the issue is the data we are passing to merge
# i guess its just bytes, but it needs to be a list of bytes ... so when it gets indexed it does weird stuff.
# lets look at old code and figure out what step we are missing, I am guessing its that split stuff.
# bpe_encoding.py

PAT_REGEX = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", re.UNICODE)
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

class Tokenizer:
  def __init__(self, vocab, merges, special_tokens=None):
    self.vocab = vocab
    self.reverse_vocab = { value: key for (key, value) in self.vocab.items() }
    self.merges = merges
    self.special_tokens = special_tokens

  def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
    assert False

  '''
  def encode(self, text: str) -> list[int]:
    print ()
    #print (self.merges[0])
    text = text.encode("utf-8")
    #print (text)
    for pair in self.merges:
      #print (text, end=' ')
      text = merge(text, pair, pair[0] + pair[1])
      #print (pair, text)
    #print (text)
    ids = []
    for token in text:
      ids.append(self.reverse_vocab[token])
    return ids
  '''

  def encode(self, text: str) -> list[int]:
    print ()

    '''
    ids = []
    for match in PAT_REGEX.finditer(text):
      word = match.group(0)
      word = word.encode("utf-8")
      for pair in self.merges:
        text = merge(text, pair, pair[0] + pair[1])
    '''

    tokens = re.findall(PAT, text)
    tokens = [ token.encode("utf-8") for token in tokens ]
    for pair in self.merges:
      tokens = merge(tokens, pair, pair[0] + pair[1])

    ids = []
    for token in tokens:
      ids.append(self.reverse_vocab[token])

    return ids

  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    pass

  def decode(self, ids: list[int]) -> str:
    ret = ''
    for id in ids:
      ret += str( self.vocab[id].decode('utf-8') )
    return ret








