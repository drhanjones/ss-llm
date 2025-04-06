# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
import time
import platform
import os


# wandb logging
dataset = 'babylm_full_bpe_100M_8k'
wandb_log = True # disabled by default
wandb_project = 'wikipedia'
sysname = "local" if "pop-os" in platform.node() else "server"
runtype = "randomseed_test" # default or random seed test
 
save_sample_to_file = True # if True, save a sample to file after each eval, overwrite in config
sampling_frequency = 10000 # how often to sample from the model, overwrite in config

wm_mask = True
wm_decay_rate = 2
wm_decay_type = "exponential_2"
wm_decay_echoic_memory = 1


# baby GPT model :)
n_layer = 12
n_head = 12

if wm_mask:
    mask_part = "mask_"
    if wm_decay_type == "exponential":
        mask_part += f"e{wm_decay_rate:03}".replace(".", "p")
    elif wm_decay_type == "exponential_2":
        mask_part += f"ee{wm_decay_rate:03}".replace(".", "p")
    elif wm_decay_type == "logarithmic":
        mask_part += f"log{wm_decay_rate:03}".replace(".", "p")
    elif wm_decay_type == "linear":
        mask_part += "lin"
    elif wm_decay_type == "inverse_sigmoid":
        mask_part += "sig"
        #convert decay rate to string (usually is value less than 1 so need to find a way to represent decimal in string)
        mask_part += f"{wm_decay_rate:03}".replace(".", "p")
    elif wm_decay_type == "custom_logistic":
        mask_part += "sig"
        #convert decay rate to string (usually is value less than 1 so need to find a way to represent decimal in string)
        mask_part += f"{wm_decay_rate:03}".replace(".", "p")

    mask_part += f"_em{wm_decay_echoic_memory:02}"

else:
    mask_part = "nomask"



lay_x_head = f'{n_layer}x{n_head}'

torch_seed_default = 1337

unique_id = os.environ.get("SLURM_JOB_ID", str(int(time.time()))) #Using SLURM_JOB_ID as unique id else use time#  #str(int(time.time()))
out_dir = f'output_dump/out-{dataset}-{lay_x_head}-{mask_part}-{unique_id}'
wandb_run_name = f'{dataset}_{lay_x_head}_{mask_part}_gpt2_{sysname}_run_{unique_id}'


if runtype == "randomseed_test":
    wandb_run_name += f'_s{torch_seed_default}'
    out_dir += f'_s{torch_seed_default}'

auto_blimp_eval = False

if auto_blimp_eval:
    #adding out_dir and data dir to shell for bash script to use

    write_path = os.environ.get("HOME")
    print("Setting up blimp eval and adding out_dir and data dir to shell for bash script to use")

    with open(f"{write_path}/blimp_out_dir", "w") as f:
        f.write(out_dir)

    with open(f"{write_path}/blimp_data_dir", "w") as f:
        f.write(dataset)


eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

gradient_accumulation_steps = 4 * 2
batch_size = 32  #64
block_size = 256


n_embd = 768

dropout = 0.1


max_iters = 44000

#Using Hyperparameters from BabyStories paper
learning_rate = 1e-5 #Default GPT2 acc to Andrej Repo is 6e-4. LR used in 6x6_10M is 5e-4
lr_decay_iters = 44000 # make equal to max_iters usually
min_lr = 1e-6 # learning_rate / 10 usually
#beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
#beta2 = 0.999 in BabyStories




# weight decay
weight_decay = 1e-1

warmup_iters = 100 # not super necessary potentially


# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model


###############################################################################################################
#  ---------------------------------  Notes on Choice of Hyperparameters  ---------------------------------  #
###############################################################################################################



################################################################################################################
# --------------------------------- Number of Iterations ------------------------------------------------------ #
################################################################################################################

# Following analysis is for calculating number of iterations

#In previous analyses, for 6 Layer 6 Head model for 10M tokens, we used 44000 iterations
#65536 tokens per iteration = 8 gradient accumulation steps * 32 batch size * 256 block size
# 44000 * 65536 = 2883584000 tokens = 2.88B tokens (~288 Epochs?)
# Data seen per iteration is 65536/10M = (in percentage) 0.655% of the data

# For context GPT-wee has 40000 steps with 32 batch size and 128 block size.
# 40000 * 32 * 128 = 163840000 tokens = 163.84M tokens (for 10M tokens, this is 16 epochs)

# Need to use these to calculate amount of data for 12x12 GPT2 model, for 100M tokens

# Option 1: Use Andrej's calculations for 12x12 GPT2 model, with OWT dataset
# OWT has 9B training tokens, 4M validation tokens #Refer -  nanoGPT/data/openwebtext/readme.md
# 40 Gradient Accumulation Steps, 12 Batch Size, 1024 Block Size = 40 * 12 * 1024 = 491520 tokens per iteration
# Data seen per iteration is 491520/9B = (in percentage) 0.0055% of the data
# Total Iterations = 600000
# Total Tokens = 600000 * 491520 = 294912000000 tokens = 294.912B tokens
# When calculated by epochs, 294.912B tokens at 9B tokens per epoch = 32 epochs of 9B token OWT data

# Same when used for 100M token dataset, 294.912B tokens = 2949 Epochs of 100M token BabyLM data
# This clearly feels overkill(??)

# Option 2: Use BabyStories's measure of 15 epochs
#BabyStories used 15 epochs (30 in codebase), with context window of 1024 tokens, batch size of 8, gradient accumulation steps of 2

# 15(30) Epochs, 100M tokens -> 1500M(3000M) tokens = 1.5B(3B) tokens
#When translated to our setup -> 1.5B tokens = 1.5B/65536 = 22,888 iterations
#                             -> 3B tokens = 3B/65536 = 45,776 iterations
# Stick to the same 44000 iterations (?)

# Option 3: Use 6x6_10M model as reference, and scale up to 12x12 GPT2 model
# If 10M dataset was run for 288 epochs, ie 44000 iterations for 2.88B tokens
# 100M dataset run for the same number of epochs would have 100M * 288 = 28.8B tokens
# 28.8B tokens at the rate of 65536 tokens per iteration
# = 28.8B/65536 = 439453 iterations
# Once again feels like overkill (???)

# Option 4: Use same data scale (288 epochs) but increase block size to 1024
#Not really recommended because then different context window for different models
# 28.8B tokens, with 1024*32*8 = 262144 tokens per iteration
# = 28.8B/262144 = 110,000 iterations


# 15 Epochs = 150M tokens, at rate of 65536 tokens per iteration,
# we need 150M/65536 = 2288 iterations, at 250 iterations per logging steps,
# we are roughly referring to 9-10 steps in wandb
# according to our original setup (Instead of 44000 iterations/ 176 steps)






#  -----------------------------------------------------------------------------
# # default config values designed to train a gpt2 (124M) on OpenWebText
# # I/O
# out_dir = 'out'
# eval_interval = 2000
# log_interval = 1
# eval_iters = 200
# eval_only = False # if True, script exits right after the first eval
# always_save_checkpoint = True # if True, always save a checkpoint after each eval
# init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
#
# # data
# dataset = 'openwebtext'
# gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
# batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
# block_size = 1024
#
# # model
# n_layer = 12
# n_head = 12
# n_embd = 768
# dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# bias = False # do we use bias inside LayerNorm and Linear layers?
# wm_mask = False # whether to use a mask for the attention layer
# wm_decay_rate = 2 # how fast to decay the mask
# wm_decay_type = "linear" # "linear" or "exponential"
#
#
# # adamw optimizer
# learning_rate = 6e-4 # max learning rate
# max_iters = 600000 # total number of training iterations
# weight_decay = 1e-1
# beta1 = 0.9
# beta2 = 0.95
# grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
#
# # learning rate decay settings
# decay_lr = True # whether to decay the learning rate
# warmup_iters = 2000 # how many steps to warm up for
# lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
# min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla


