
import torch
import numpy as np
import matplotlib.pyplot as plt
from cs336_basics import model

##################################

# GPT-2
m = model.TransformerDecoder(vocab_size=50257, context_length=1024, d_model=1600, num_layers=48, num_heads=25, d_ff=6400, rope_theta=10000)

# Small
# m = model.TransformerDecoder(vocab_size=50257, context_length=1024, d_model=1600, num_layers=2, num_heads=25, d_ff=6400, rope_theta=10000)

# print (m)

##################################

# inputs = torch.zeros([1, 69], dtype=torch.int32)
inputs = torch.zeros([1, 1], dtype=torch.int32)
out = m.forward(inputs)

flops = m.count_flops()
breakdown = { 'attn': 0, 'ffn': 0, 'norm': 0 }
for key, value in flops.items():
  print (key, value)
  if 'attn' in key:   breakdown['attn'] += value
  elif 'Norm' in key: breakdown['norm'] += value
  else:               breakdown['ffn'] += value

print ('---------------------')

print ('Total Paramters (B)', m.count_params() / 1e9)
print ('FFN Flops (B)', breakdown['ffn'] / 1e9)
print ('Attention Flops (B)', breakdown['attn'] / 1e9)
print ('Norm Flops (B)', breakdown['norm'] / 1e9)

##################################
