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

text = '''
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
'''

lut = {}
for word in text.split():
  word = word.encode("utf-8")
  word_bytes = tuple(bytes([x]) for x in word)
  if word_bytes not in lut.keys(): lut[word_bytes] = 0
  lut[word_bytes] += 1

#print (lut)

counts = {}
for word in lut.keys():
  for b1, b2 in zip(word, word[1:]):
    if (b1, b2) not in counts.keys(): counts[(b1, b2)] = 0
    counts[(b1, b2)] += lut[word]

print (counts)
#pair = max(counts, key=counts.get)
pair = max(counts.items(), key=lambda x: (x[1], x[0]))
print (pair)
