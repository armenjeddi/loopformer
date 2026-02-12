# wandb_log = True
# wandb_project = 'owt'
# wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 50000
lr_decay_iters = 50000

out_dir = "runs/loopformer_3blk"
dataset = "pile_deduplicated"

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 2e-1

### model cfg
model_type: str = 'loopformer'
max_model_loops: int = 8
n_layer: int = 3
n_head: int = 32
n_embd: int = 2048
