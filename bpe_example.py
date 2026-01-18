#####################

def merge(indices, pair, new_index):
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

vocab = { x: bytes([x]) for x in range(256) }
vocab['<|endoftext|>'] = 256

#####################

# so the new problem is that we have to apply merge each time through on all text.
# this is explained on page 8.

lut = {}
for word in text.split():
  word = word.encode("utf-8")
  indices = tuple(map(int, word))
  if indices not in lut.keys():
    lut[indices] = 0
  lut[indices] += 1

#####################

for step in range(12):
  counts = {}
  for indices in lut.keys():
    for pair in zip(indices, indices[1:]):
      if pair not in counts.keys():
        counts[pair] = 0
      counts[pair] += lut[indices]

  (pair, _) = max(counts.items(), key=lambda x: (x[1], x[0]))
  index1, index2 = pair
  new_index = len(vocab)
  vocab[new_index] = vocab[index1] + vocab[index2]

  new_lut = {}
  for (indices, count) in lut.items():
    indices = merge(indices, pair, new_index)
    new_lut[tuple(indices)] = count
  lut = new_lut

  print ( list(vocab[x] for x in pair) )

  #####################



