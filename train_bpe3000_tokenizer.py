from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize a BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]")) #

# Define a pre-tokenizer (e.g., splitting on whitespace)
tokenizer.pre_tokenizer = Whitespace()

# Initialize the trainer with desired vocabulary size and special tokens
trainer = BpeTrainer(vocab_size=3000, min_frequency=2, special_tokens=["<|endoftext|>"])

tokenizer.train(['./data/TinyStoriesV2-GPT4-train.txt'], trainer=trainer)

tokenizer.save("bpe3000.json")

#############################################

from transformers import PreTrainedTokenizerFast

# Load the custom tokenizer from the saved JSON file
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe3000.json") #

# Test encoding some text
output = hf_tokenizer.encode("Hello world, this is my custom tokenizer.")
print(output)

#############################################
