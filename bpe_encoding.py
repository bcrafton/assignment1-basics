
import regex as re

text = 'the cat ate'
vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'),(b' a', b't')]

vocab_reverse = { value: key for (key, value) in vocab.items() }

PAT_REGEX = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", re.UNICODE)

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

def encocde(word, merges):
  '''
  for pair in zip(word, word[1:]):
    print (pair)
  '''
  for pair in merges:
    word = merge(word, pair, pair[0] + pair[1])
  print (word)
  for char in word:
    print(vocab_reverse[char])
  return word

for match in PAT_REGEX.finditer(text):
  word = match.group(0)
  word = word.encode("utf-8")
  word_bytes = tuple(bytes([x]) for x in word)
  #print (word_bytes)
  encocde(word_bytes, merges)
