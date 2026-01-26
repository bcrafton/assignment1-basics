
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
    if special_tokens:
      self.special_tokens = special_tokens
    else:
      self.special_tokens = []
    '''
    if self.special_tokens:
      self.split_pattern = re.compile("|".join(re.escape(st) for st in special_tokens))
    else:
      self.split_pattern = None
    '''

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

    # okay I think its obvious we need to go back and study this: "2.2 Unicode Encodings"
    # and we probably also need to understand the 2 regex:
    # 1- PAT_REGEX
    # 2- re.compile("|".join(re.escape(st) for st in special_tokens))

    # so the problem is it isnt splitting the bytes into a list
    # but i dont think its the unicode smileys fault.

    # yeah so i think we need to do this:
    # word_bytes = tuple(bytes([x]) for x in word)

    # check unicode_test.py

    # this is the test:
    # test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
    # encoded_ids = tokenizer.encode(test_string)
    # tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
    # # Ensure the special <|endoftext|> token is preserved
    # assert tokenized_string.count("<|endoftext|>") == 3

    # so it actually wants us to preserve the special token

    tokens = []

    if self.special_tokens:
      special_pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
      parts = re.split(special_pattern, text)
    else:
      parts = [text]

    for part in parts:
      if part in self.special_tokens:
        tokens.append(part.encode("utf-8"))
      else:
        words = re.findall(PAT, part)
        for word in words:
          word = word.encode("utf-8")
          tokens.extend(list(bytes([x]) for x in word))

    for pair in self.merges:
      tokens = merge(tokens, pair, pair[0] + pair[1])

    ids = []
    for token in tokens:
      ids.append(self.reverse_vocab[token])

    return ids

  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    pass

  def decode(self, ids: list[int]) -> str:
    ret = b''
    for id in ids:
      ret += self.vocab[id]
    ret = str( ret.decode('utf-8', errors='replace') )
    return ret







