import os
import requests
import tiktoken
import numpy as np
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer)
from transformers import PreTrainedTokenizerFast
from tokenizers.normalizers import Lowercase, Strip, StripAccents, NFD
from transformers import AutoTokenizer

from pathlib import Path
import pickle

dataset_folder_path = Path(__file__).parents[3] / 'babylm_data'
train_ds = 'babylm_100M'
dev_ds = 'babylm_dev'
dataset_name = "full"
vocabulary_size = 8000

textfiles = [f"{dataset_folder_path}/{train_ds}/aochildes.train",
             f"{dataset_folder_path}/{train_ds}/bnc_spoken.train",
             f"{dataset_folder_path}/{train_ds}/cbt.train",
             f"{dataset_folder_path}/{train_ds}/children_stories.train",
             f"{dataset_folder_path}/{train_ds}/gutenberg.train",
             f"{dataset_folder_path}/{train_ds}/open_subtitles.train",
             f"{dataset_folder_path}/{train_ds}/qed.train",
             f"{dataset_folder_path}/{train_ds}/simple_wikipedia.train",
             f"{dataset_folder_path}/{train_ds}/switchboard.train",
             f"{dataset_folder_path}/{train_ds}/wikipedia.train"]





if dataset_name == "full":
    train_file_path = [ os.path.join(dataset_folder_path, train_ds, x) for x in os.listdir(os.path.join(dataset_folder_path, train_ds)) if x.endswith(".train")]
    dev_file_path = [ os.path.join(dataset_folder_path, dev_ds, x) for x in os.listdir(os.path.join(dataset_folder_path, dev_ds)) if x.endswith(".dev")]


else:
    train_file_path = [os.path.join(dataset_folder_path, train_ds, dataset_name + '.train')]
    dev_file_path = [os.path.join(dataset_folder_path, dev_ds, dataset_name + '.dev')]

#read all the files from the list of paths given

train_data = []
for file in train_file_path:
    t_file = ""

    with open(file, 'r') as f:
        t_file += f.read()

        #Add new line to the end of each file

        t_file += "\n"

    train_data.append(t_file)

val_data = []

for file in dev_file_path:
    v_file = ""
    with open(file, 'r') as f:
        v_file += f.read()

        #Add new line to the end of each file

        v_file += "\n"

    val_data.append(v_file)


# with open(train_file_path, 'r') as f:
#     train_data = f.read()

# with open(dev_file_path, 'r') as f:
#     val_data = f.read()





meta = {}

def encode_with_tokenizer(data, textfiles,  tokenizer_type = "customBPE", vocab_size = None, tokenizer_scratch = False, traintokenizer_1ds = False):

    if tokenizer_type == "customBPE":
        # BUILD A CUSTOM BPE TOKENIZER
        meta["custom_tokenizer"] = True

        #Check if tokenizer already exists

        tokenizer_exists = True if os.path.exists("tokenizer.json") else False

        if tokenizer_scratch or not tokenizer_exists:

            tokenizer = Tokenizer(models.BPE())
            tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), Strip(), StripAccents()])
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])

            if traintokenizer_1ds:
                textfiles = [train_file_path]

            tokenizer.train(files=textfiles, trainer=trainer)

            tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
            tokenizer.decoder = decoders.ByteLevel()

            # Save the tokenizer
            wrapped_tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                bos_token="<|endoftext|>",
                eos_token="<|endoftext|>")

            print("Saving tokenizer to", Path(__file__).parents[0])
            wrapped_tokenizer.save_pretrained(Path(__file__).parents[0])
            tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parents[0])

        else:
            print("Tokenizer already exists. Loading from file...")
            tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parents[0])


        print("Encoding data with tokenizer...")
        data_ids = np.concatenate(tokenizer(data, return_tensors="np")["input_ids"])
        print("Encoding done.")
        meta["vocab_size"] = tokenizer.vocab_size

    elif tokenizer_type == "gpt2":
        meta["custom_tokenizer"] = False
        enc = tiktoken.get_encoding("gpt2")
        data_ids = enc.encode_ordinary(data)

    return data_ids

print("Encoding train and val data...")
#print("Sample train data:", repr(train_data[:1500]))
#print("Sample val data:", repr(val_data[500:1500]))
import time
# encode with tiktoken gpt2 bpe
start = time.time()
train_ids = encode_with_tokenizer(train_data, textfiles, tokenizer_type = "customBPE", vocab_size = vocabulary_size)
print("Time taken to encode train data:", time.time() - start)

start = time.time()
val_ids = encode_with_tokenizer(val_data, textfiles, tokenizer_type =  "customBPE", vocab_size = vocabulary_size)
print("Time taken to encode val data:", time.time() - start)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)


train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# When using GPT tokenizer:
# train has 1,377,181 tokens
# val has 1,594,771 tokens


# When using custom BPE tokenizer:

# train has 1,426,302 tokens
# val has 1,667,685 tokens


#When using full data of babylm with custom BPE tokenizer and 16000 vocab size

# train has 13,636,233 tokens
# val has 13,077,391 tokens

#When using full data of babylm with custom BPE tokenizer and 8000 vocab size

# Tokenizer vocab size: 8000
# train has 14,545,185 tokens
# val has 13,921,894 tokens
