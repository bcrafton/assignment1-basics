
from cs336_basics.train_bpe_tokenizer_parallel import train_bpe_parallel

train_bpe_parallel('./data/TinyStoriesV2-GPT4-train.txt', 10000, '<|endoftext|>')
