"""
Sample from a trained model

To sample
$ python sample.py config/sample_config.py --out_dir=list_output_dir

python sample --out_dir=output_dump/out-babylm_full_bpe-4x4-nomask-5444724 config/babylm_full_bpe/train_blmf_gpt_nomask.py
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from transformers import AutoTokenizer
import sys
from tokenizers import decoders

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "Thee" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 3 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dataset = "openwebtext" # placeholder?
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

#Manually overriding just the outdir because of naming issues

for arg in sys.argv[1:]:
    if '=' in arg:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]

        if key == "out_dir":
            print(f"Overriding: {key} = {val}")
            print(f"Overriding out_dir with {val}")
            out_dir = val



torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    print("G2",gptconf)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes

    #CHECK IF CUSTOM BPE IS AN OPTION IN META

    if meta.get("custom_tokenizer", False):
        print("Using custom tokenizer from meta.pkl")

        #tokenizer_path = os.path.join(data_dir, 'tokenizer.json')
        #Ideally you'll want to save the tokenizer and read that file
        #Maybe save file name of tokenizer and then load it here

        #GET PATH OF DATASET FROM CONFIG

        data_dir = os.path.join('data', dataset)
        custom_tokenizer = AutoTokenizer.from_pretrained(data_dir)

        #Probably add below line to the prepare.py when you are creating a tokenizer and then try to see if it works
        #custom_tokenizer.decode = decoders.ByteLevel()


        encode = lambda s: custom_tokenizer.encode(s)
        decode = lambda l: custom_tokenizer.decode(l)
        #Decode but replace that special character with a space manually
        #decode = lambda l: "".join(custom_tokenizer.decode(l).split("Ġ"))

        #encode = lambda s: meta['tokenizer'].encode(s)
        #decode = lambda l: meta['tokenizer'].decode(l)
    else:
        if meta.get("stoi", False):
            print("Using stoi/itos from meta.pkl")
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
        else:
            pass
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

print("Prompt:", repr(start))

start_ids = encode(start)

print("start_ids",start_ids)

x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)

            print(decode(y[0].tolist()))
            print('---------------')
