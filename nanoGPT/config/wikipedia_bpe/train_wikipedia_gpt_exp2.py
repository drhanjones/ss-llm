# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
import time 


# wandb logging
dataset = 'wikipedia_bpe'
wandb_log = True # disabled by default
wandb_project = 'wikipedia'

wm_mask = True
wm_decay_rate = 2
wm_decay_type = "exponential"


if wm_mask:
    mask_part = "mask_"
    if wm_decay_type == "exponential":
        mask_part += f"e{wm_decay_rate:03}"
    elif wm_decay_type == "linear":
        mask_part += "lin"
else:
    mask_part = "nomask"

out_dir = f'output_dump/out-{dataset}-{mask_part}'
wandb_run_name = f'{dataset}_{mask_part}_gpt2'+ "run" + str(int(time.time()))


#out_dir = 'out-wikipedia-char-mask'
#wandb_run_name = 'wikipedia_char_mask_e500_gpt2'+ "run" + str(int(time.time()))

#out_dir, wandb_run_name = get_save_name(dataset, wm_mask, wm_decay_rate, wm_decay_type)

eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

gradient_accumulation_steps = 8
batch_size = 32  #64
block_size = 128

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 256

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

#$ python train.py config/train_shakespeare_char_test.py





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


