# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
import time 


# wandb logging
dataset = 'wikipedia_char'
wandb_log = True # disabled by default
wandb_project = 'wikipedia'

wm_mask = True
wm_decay_rate = 10
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

#wandb_log = False # override via command line if you like
#wandb_project = 'shakespeare-char'
#wandb_run_name = 'mini-gpt'






gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

#$ python train.py config/train_shakespeare_char_test.py