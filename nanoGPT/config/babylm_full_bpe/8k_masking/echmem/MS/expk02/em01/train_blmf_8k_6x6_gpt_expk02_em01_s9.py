# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
import time
import platform
import os


# wandb logging
dataset = 'babylm_full_bpe_8k'
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
n_layer = 6
n_head = 6

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

torch_seed_default = 11111

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

gradient_accumulation_steps = 8
batch_size = 32  #64
block_size = 256


n_embd = 384

dropout = 0.1

learning_rate = 5e-4 #1e-3 # with baby networks can afford to go a bit higher
max_iters = 44000
lr_decay_iters = 44000 # make equal to max_iters usually
min_lr = 5e-5 # learning_rate / 10 usually
#beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# weight decay
weight_decay = 1e-1

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model







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


