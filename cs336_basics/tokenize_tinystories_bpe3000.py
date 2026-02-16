
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

from transformers import PreTrainedTokenizerFast

################################

tokenizer = PreTrainedTokenizerFast(tokenizer_file="../bpe3000.json")

fr = open('/home/brian/Desktop/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt', 'r')
fw = np.memmap('tmp.dat', dtype=np.uint16, mode='w+', shape=10**9)

ptr = 0
count = 0
while True:
  text = fr.read(64*1024*1024)
  count += 1
  print (count*64*1024*1024 / 2227753162 * 100)

  if text:
    tokens = tokenizer(text, return_tensors="np")['input_ids'].reshape(-1)
    fw[ptr:ptr+len(tokens)] = tokens
    fw.flush()
    ptr += len(tokens)
    # print (ptr, ptr+len(tokens))
    # print (fw[ptr])
  else:
    break

fr.close()
del fw

################################

fr = np.memmap('tmp.dat', dtype=np.uint16, mode='r+')
fw = np.memmap('TinyStoriesV2-GPT4-train-bpe3000.dat', dtype=np.uint16, mode='w+', shape=ptr)

for i in range(ptr):
  fw[i] = fr[i]

del fr
del fw

################################



