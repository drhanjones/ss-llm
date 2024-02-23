import os
import requests
import tiktoken
import numpy as np


from pathlib import Path

dataset_folder_path = Path(__file__).parents[3] / 'babylm_data'
train_ds = 'babylm_10M'
dev_ds = 'babylm_dev'
dataset_name = "wikipedia"

train_file_path = os.path.join(dataset_folder_path, train_ds, dataset_name + '.train')
dev_file_path = os.path.join(dataset_folder_path, dev_ds, dataset_name + '.dev')

train_file_path = os.path.join(dataset_folder_path, train_ds, dataset_name + '.train')
dev_file_path = os.path.join(dataset_folder_path, dev_ds, dataset_name + '.dev')

with open(train_file_path, 'r') as f:
    train_data = f.read()
#print(f"length of train dataset in characters: {len(train_data):,}")

with open(dev_file_path, 'r') as f:
    val_data = f.read()
#print(f"length of val dataset in characters: {len(val_data):,}")

meta = {}

def encode_with_tokenizer(data, tokenizer = "customBPE", vocab_size = None):

    if tokenizer == "customBPE":
        # BUILD A CUSTOM BPE TOKENIZER
        meta["custom_tokenizer"] = True
        pass

    elif tokenizer == "gpt2":
        meta["custom_tokenizer"] = False
        enc = tiktoken.get_encoding("gpt2")
        data_ids = enc.encode_ordinary(data)

    return data_ids


# encode with tiktoken gpt2 bpe
train_ids = encode_with_tokenizer(train_data, tokenizer = "gpt2")
val_ids = encode_with_tokenizer(val_data, tokenizer =  "gpt2")

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)


train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train has 1,377,181 tokens
# val has 1,594,771 tokens
