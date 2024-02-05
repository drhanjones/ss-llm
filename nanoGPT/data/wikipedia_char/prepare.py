"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np
import string
from pathlib import Path
# # download the tiny shakespeare dataset
# input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# if not os.path.exists(input_file_path):
#     data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
#     with open(input_file_path, 'w') as f:
#         f.write(requests.get(data_url).text)

dataset_folder_path = Path(__file__).parents[3] / 'babylm_data'
#dataset_folder_path = r'/ss-llm/babylm_data'

train_ds = 'babylm_10M'
dev_ds = 'babylm_dev'
dataset_name = "wikipedia"

train_file_path = os.path.join(dataset_folder_path, train_ds, dataset_name + '.train')
dev_file_path = os.path.join(dataset_folder_path, dev_ds, dataset_name + '.dev')

with open(train_file_path, 'r') as f:
    train_data = f.read()
print(f"length of train dataset in characters: {len(train_data):,}")

with open(dev_file_path, 'r') as f:
    val_data = f.read()
print(f"length of val dataset in characters: {len(val_data):,}")

def clean_text(text):

    # Remove characters are not ascii
    text = ''.join(filter(lambda x: x in string.printable, text))
    return text

train_data = clean_text(train_data)
val_data = clean_text(val_data)



# get all the unique characters that occur in this text
data = train_data + val_data
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }


def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
#n = len(data)
#train_data = data[:int(n*0.9)]
#val_data = data[int(n*0.9):]

# encode both to integers

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens

"""
length of train dataset in characters: 9,087,222
length of val dataset in characters: 9,396,525
all the unique characters: 
 !"#$%&'()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~
vocab size: 94
train has 9,069,201 tokens
val has 9,376,800 tokens




length of train dataset in characters: 6,065,862
length of val dataset in characters: 7,007,380
all the unique characters: 
 !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~
vocab size: 97
train has 6,056,937 tokens
val has 6,995,039 tokens
"""