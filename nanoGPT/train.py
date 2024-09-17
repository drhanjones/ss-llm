"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""


import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT   #, WMConfig

from utils.sampler import sample_from_model


# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())

torch_seed_default = 1337

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

if init_from == 'resume':
    wandb_run_id = None #Wandb expect run id from config when resume else fail

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# experimenting with head size in qkv matrices
head_size_qkv = None
ffw_dim = None

wm_mask = False # whether to use a mask for the attention layer
wm_decay_length = None #setting to none and after configurator is run, if config doesn't provide a value, set to block_size
wm_decay_rate = 2 # how fast to decay the mask
wm_decay_type = "linear" # Type of decay to apply to the matrix #linear, exponential, inverse_sigmoid, custom_logistic
wm_decay_echoic_memory = 1 #Echoic memory for the decay matrix, first n values where "effect of decay" is not applied, where memory is supposedly perfect
wm_setting_type = "old" # "old" or "new # old is the original implementation where mask is applied after softmax, new is where mask is applied before softmax


# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
#Alternatively, use epochs if dataset type is 'full'
epochs = 10

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# Variable for batchloader - randomised or full data set

batchloader = 'random' # 'random' or 'full'

#SAMPLING Settings

save_sample_to_file = False # if True, save a sample to file after each eval, overwrite in config
sampling_frequency = 5000 # how often to sample from the model, overwrite in config

#Curriculum Settings
curriculum = False # if True, use curriculum learning
curriculum_type = 'data' # 'data' or 'objective' or 'vocab'
curriculum_pacing_fn_name = "log" # "linear", "quad", "root", "log", "exp", 'step'
curriculum_start_difficulty = 0.2 # starting difficulty of the curriculum what is the default space to sample from, maybe overwrite dynamically based on pacing function
curriculum_max_difficulty = 1.0 #Keep as 1
curriculum_start_percent = 0.2 #When does the curriculum start increasing the difficulty
curriculum_end_percent = 0.8 #When does the curriculum reach maximum difficulty
pacing_percentile_to_max_impl = "default" # default or absolute, where absolute it just percentile * max_difficulty while default is converts percentile to position in sorted list of values
curriculum_log_growth_rate_c = 10 #Growth rate for the log function, only used if pacing function is log

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(torch_seed_default + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
print(f"loading data from {data_dir}")
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

from utils.curriculum_dataloader import dataloader as cdl
from utils.tokenizer_utils import load_tokenizer

if curriculum:

    tokenizer = load_tokenizer(data_dir)
    #Overwriting train and val data with curriculum data
    train_data, val_data, train_data_indices, val_data_indices, pacing_fn = cdl.get_cc_dataset_and_pacing_fn(
        tokenizer=tokenizer, block_size=block_size, total_steps=max_iters, spoken_first=True,
        pacing_fn_name=curriculum_pacing_fn_name, start_percent=curriculum_start_percent,
        end_percent=curriculum_end_percent, starting_difficulty=curriculum_start_difficulty,
        max_difficulty=curriculum_max_difficulty, growth_rate_c=curriculum_log_growth_rate_c)

    train_curriculum_difficulties = np.array(train_data["curriculum_order"])
    val_curriculum_difficulties = np.array(val_data["curriculum_order"])
    print("train_data_indices: ", train_data_indices)
    print("val_data_indices: ", val_data_indices)
    print(train_curriculum_difficulties.shape)
    print("Curriculum data loaded successfully")
    print("starting max difficulty is: ", int(np.percentile(train_curriculum_difficulties, pacing_fn(0)*100)))


def get_batch(split, batchloader='random', batch_idx = 0, current_iter = None):

    data = train_data if split == 'train' else val_data
    if batchloader == 'random':
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        # if device_type == 'cuda':
        #     # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        #     x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        # else:
        #     x, y = x.to(device), y.to(device)
        # return x, y

    elif batchloader == 'full':
        raise NotImplementedError("Full batch loader not implemented yet")
        pass

    elif batchloader == 'curriculum':
        if current_iter is None:
            raise ValueError("Current iteration is not provided for curriculum batch loader, required for pacing function")

        data = train_data if split == 'train' else val_data
        indices_dict = train_data_indices if split == 'train' else val_data_indices
        #difficulties_list = train_curriculum_difficulties #if split == 'train' else val_curriculum_difficulties
        #No "if" I think because it should be using the train difficulties to determine the pacing score
        #PROBABLY NEED TO VERIFY THIS LOGIC
        pacing_difficulty = cdl.convert_pacing_fn_to_max_difficulty(pacing_fn, current_iter, train_curriculum_difficulties, implementation_type=pacing_percentile_to_max_impl)
        #print("Pacing difficulty is: ", pacing_difficulty)
        #Get the indices of the data that are within the pacing difficulty
        if split == 'train':
            ix = torch.randint(indices_dict[pacing_difficulty], (batch_size,))
        else:
            #just use entire data for validation
            ix = torch.randint(len(data), (batch_size,))

        data_rows = data.select(ix) #data is type of datasets so select is used to get the rows
        x = torch.tensor(data_rows["input_ids"])
        y = torch.tensor(data_rows["output_ids"])

        # #Assert shapes of x and y
        # assert x.shape == (batch_size, block_size)
        # assert y.shape == (batch_size, block_size)

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y



# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size', None)
    if meta_vocab_size:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

#Setting wm_decay_length TO BLOCK SIZE if it is not set already
if wm_decay_length is None:
    wm_decay_length = block_size



# Setting head size as 3 times n_embd if not set already
if head_size_qkv is None:
    head_size_qkv = n_embd

# Setting ffw_dim as 4 times n_embd if not set already
if ffw_dim is None:
    ffw_dim = 4 * n_embd

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, head_size_qkv=head_size_qkv, ffw_dim=ffw_dim,
                  wm_mask=wm_mask, wm_decay_rate=wm_decay_rate, wm_decay_type=wm_decay_type,
                  wm_decay_length=wm_decay_length, wm_decay_echoic_memory=wm_decay_echoic_memory,
                  wm_setting_type=wm_setting_type) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf) 
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    print("continuing from iteration", iter_num)
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

if wm_mask:
    log_decay_vals = model.transformer.h[0].attn.get_decay_weight_matrix(block_size, wm_decay_length,
                                                                     decay_factor=wm_decay_rate,
                                                                     decay_type=wm_decay_type,
                                                                     decay_echoic_memory=wm_decay_echoic_memory)

    log_decay_vals = log_decay_vals[-1].flip([0]).tolist()

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if batchloader == 'random':
                X, Y = get_batch(split)
            elif batchloader == 'full':
                pass
            elif batchloader == 'curriculum':
                X, Y = get_batch(split, batchloader='curriculum', current_iter=iter_num)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    if init_from == 'resume':
        wandb.init(project=wandb_project, name=wandb_run_name, id = wandb_run_id, resume="must", config=config)

    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

        #logging the decay curve if it exists once
        if wm_mask:
            print("Logging decay curve")
            x = list(range(len(log_decay_vals)))
            wandb.log(
                {
                    "decay_curve": wandb.plot.line(
                        wandb.Table(data=[[x,y] for x,y in zip(x, log_decay_vals)], columns=["index", "decay_value"]),
                        "index", "decay_value",
                        title=f"Decay Curve for {wm_decay_type} decay with rate {wm_decay_rate} and echoic memory {wm_decay_echoic_memory}"
                    )
                }, commit=False
            )
            #log it without using a step
            print("Logged decay curve")


# training loop
if batchloader == 'random':
    X, Y = get_batch('train') # fetch the very first batch
elif batchloader == 'full':
    pass
elif batchloader == 'curriculum':
    X, Y = get_batch('train', batchloader='curriculum', current_iter=iter_num)

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        lt1 = time.time()
        losses = estimate_loss()
        lt3 = time.time()
        print(f"Time taken for evaluation of loss: {lt3 - lt1}")
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            if not curriculum:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
            else:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                    "pacing percentile": pacing_fn(iter_num),
                    "difficulty level": cdl.convert_pacing_fn_to_max_difficulty(pacing_fn,
                                                                                iter_num,
                                                                                train_curriculum_difficulties,
                                                                                implementation_type=pacing_percentile_to_max_impl),
                    })

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        lt2 = time.time()
        print(f"Time taken for evaluation and saving checkpoint: {lt2 - lt1}")
        #Consider running a sampler here maybe?

        if save_sample_to_file:
            #Save final iter model too
            if iter_num % sampling_frequency == 0 or iter_num == max_iters:

                # sample from the model
                #Call function to sample from model and save to file
                save_start_time = time.time()
                sample_from_model(raw_model, data_dir, out_dir, dtype, device=device, prompt_type="elaborate", iter_num = iter_num)
                print("sampling from mode and saving to file in iteration ", iter_num)
                print("Time taken to sample and save to file: ", time.time() - save_start_time)


        #Add Any other logging here

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        if batchloader == 'random':
            X, Y = get_batch('train')
        elif batchloader == 'full':
            pass
        elif batchloader == 'curriculum':
            X, Y = get_batch('train', batchloader='curriculum', current_iter=iter_num)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

# %%
