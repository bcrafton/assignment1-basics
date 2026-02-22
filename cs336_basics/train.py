
import os
import json

# from train_bpe_tokenizer import *
from tokenizer import Tokenizer
import model

from nn_utils import cross_entropy, gradient_clipping
from optimizer import AdamW, get_lr_cosine_schedule
from trainer import get_batch, load_checkpoint, save_checkpoint

import numpy as np

from transformers import AutoTokenizer
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import time

from transformers import PreTrainedTokenizerFast

################################

def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

################################

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

################################
'''
def load_dataset():
  with open(FIXTURES_PATH / "tinystories_sample.txt") as f:
      corpus_contents = f.read()
  ids = tokenizer.encode(corpus_contents)
  assert tokenizer.decode(ids) == corpus_contents
'''
################################

class Trainer:
  def __init__(self):
    self.vocab_size = 50257
    self.vocab_size = 3000
    self.context_length = 64
    self.d_model = 512
    self.num_layers = 8
    self.num_heads = 8
    self.d_ff = 512
    self.rope_theta = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print (device)

    m = model.TransformerDecoder(
    vocab_size=self.vocab_size,
    context_length=self.context_length,
    d_model=self.d_model,
    num_layers=self.num_layers,
    num_heads=self.num_heads,
    d_ff=self.d_ff,
    rope_theta=self.rope_theta
    )
    
    optimizer = AdamW(m.parameters(), lr=1e-4)

    # self.tokenizer = get_tokenizer_from_vocab_merges_path('tokenizer/gpt2_vocab.json', 'tokenizer/gpt2_merges.txt', 'special.txt')
    # self.dataset = None

    # so we need to figure out how to load the dataset and stream it.
    # wasnt one of the tests running the whole tinystories thing? do they load the whole dataset there?
    # i think we commented that one out. otherwise I guess we should read the notes ...
    # here is a good example:
    # /home/brian/Desktop/cs336-assignment1-basics/cs336_basics/train.py
    # /home/brian/Desktop/cs336-assignment1-basics/cs336_basics/utils/data.py
    # to use np.mmap, we need to have the dataset as integers already ... so we need to already tokenize the data?

    # here is how they do it:
    # /home/brian/Desktop/cs336-assignment1-basics/cs336_basics/scripts/tokenize_tinystories.py
    
    # yeah so they need to run: train_bpe_tinystories.py
    # to produce: tinystories_vocab.json, tinystories_merges.txt
    # then run: tokenize_tinystories.py
    # to produce: train.bin, val.bin
    # but it breaks because of oom error.

    # so tomorrow I guess we need to use our own scripts to produce these files.
    
    # I want to load the trained bpe from gpt2 instead of trying to create our own.
    # Currently we get OOM error when we run: train_bpe_tokenizer_parallel.py

    # we just used pytorch's tokenizer and its way faster.

    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="../bpe3000.json")

    #fr = np.memmap('TinyStoriesV2-GPT4-train.dat', dtype=np.uint16, mode='r+')
    fr = np.memmap('TinyStoriesV2-GPT4-train-bpe3000.dat', dtype=np.uint16, mode='r+')

    for _ in range(10000):
      t1 = time.time()
      x, y = get_batch(dataset=fr, batch_size=256, context_length=self.context_length, device=device)
      t2 = time.time()

      x = x.int()
      y = y.long()
      p = m(x)

      # print (x)
      # print (y)

      # print (p.shape)
      # print (y.shape)
      loss = cross_entropy(p, y)
      print (loss)

      # I guess now we have to get the gradients and update our model?
      # look at train.py from cs336-assignment1-basics

      loss = loss.backward()
      # gradient_clipping(model.parameters(), 1.0)
      optimizer.step()
      t3 = time.time()

      # y = y.detach().numpy()
      # decoded_text = tokenizer.decode(y[0], skip_special_tokens=True)
      # print (decoded_text)

      p = p.cpu().detach().numpy()
      # print (p.shape)
      p = np.argmax(p, axis=-1)
      # print (p.shape)
      # decoded_text = tokenizer.decode(p[0], skip_special_tokens=True)

      print (t2 - t1, t3 - t2)
      print (tokenizer.decode(y[0], skip_special_tokens=True))
      print (tokenizer.decode(p[0], skip_special_tokens=True))
      print ()

      # flops = print (m.count_flops())
      total_flops = sum( value for value in m.count_flops().values() )
      print (total_flops)
      # have to consider that we are also running backprop and updating the weights, if it was just inference it would be higher.
      # RTX5070 --> 30 TFLOPS

trainer = Trainer()

################################



