"""
Evaluate next-token-prediction perplexity of pre-trained model
"""

import gc
from model_speculative import GPT
from model_quantized import GPT_Q
from contextlib import nullcontext
import numpy as np
import os
import tiktoken
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm

# -----------------------------------------------------------------------------
init_from = 'gpt2' # a gpt2 variant (e.g. 'gpt2-xl')
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
dataset = 'wikitext' # 'shakespeare' or 'wikitext'
block_size = 1024
num_warmup = 1 # how many warmups to do before benchmarking
max_new_tokens = 50 # number of tokens generated in each sample
temperature = 0.4 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
speculative_tokens = 3 # how many tokens should the draft model decode?
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
# -----------------------------------------------------------------------------

# Setup environment
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Initialize tokenizer
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# Load dataset
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Measure perplexity on dataset
# https://huggingface.co/docs/transformers/perplexity
# Table 3, https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
def measure_perplexity(model, data, batch_size):
    nll_weights = []
    nlls = []
    for i in range(0, len(data), block_size * batch_size):
        j = min(i + block_size * batch_size, len(data))
        ix = torch.arange(i, j, block_size)
        x = []
        for k in ix:
            x.append(torch.from_numpy((data[k:k+block_size]).astype(np.int64)))
        x = torch.stack([F.pad(y, (0, block_size - len(y)), value=-1) for y in x])
        nll_weights.append((x != -1).sum().item() / len(data))
        if device_type == 'cuda':
            # pin array x which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        with torch.no_grad():
            with ctx:
                # y = x[:, 1:].clone()
                # x[x == -1] = 0
                # logits, loss = model(x[:, :-1], y)
                # nlls.append(loss)
                # https://github.com/huggingface/transformers/blob/df5c5c62ae253055336f5bb0828ca8e3e15ab6bd/src/transformers/models/gpt2/modeling_gpt2.py#L1099
                y = x.clone()
                x[x == -1] = 0
                logits, _ = model(x, y)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = y[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-1)
                nlls.append(loss)
    nlls = [nll_weights[i] * nlls[i] for i in range(len(nlls))]
    return torch.exp(torch.stack(nlls).sum()).item()

# ------------------------------------------------------------------------------
# Deliverables 1 and 2
# ------------------------------------------------------------------------------
print("-" * 80 + "\nInference: without quantization\n" + "-" * 80)
# Load pre-trained model
model_pt = GPT.from_pretrained(init_from, dict(dropout=0.0))
model_pt.eval()
torch.cuda.reset_peak_memory_stats(device=device)
model_pt.to(device)
print(f"\nGPU memory allocated after calling model.to(device) {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
if compile:
    model_pt = torch.compile(model_pt) # requires PyTorch 2.0 (optional)
# Run evaluation
torch.cuda.reset_peak_memory_stats(device=device)
ppl_pt_bs4 = measure_perplexity(model_pt, val_data, batch_size=4)
print(f"GPT-2 perplexity on {dataset}/val.bin, batch_size={4}: {ppl_pt_bs4:.4f}")
print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
print()
ppl_pt_bs12 = measure_perplexity(model_pt, val_data, batch_size=12)
print(f"GPT-2 perplexity on {dataset}/val.bin, batch_size={12}: {ppl_pt_bs12:.4f}")
print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
del model_pt
gc.collect()
torch.cuda.empty_cache()
print("\n")

print("-" * 80 + "\nInference: with quantization\n" + "-" * 80)
# Load pre-trained model and quantize
model_dq = GPT_Q.from_pretrained(init_from, dict(dropout=0.0))
model_dq.quantize_all_parameters()
torch.cuda.reset_peak_memory_stats(device=device)
model_dq.to(device)
print(f"\nGPU memory allocated after calling model.to(device) {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
if compile:
    model_dq = torch.compile(model_dq) # requires PyTorch 2.0 (optional)
# Run evaluation
torch.cuda.reset_peak_memory_stats(device=device)
ppl_dq_bs4 = measure_perplexity(model_dq, val_data, batch_size=4)
print(f"GPT-2 quantized perplexity on {dataset}/val.bin, batch_size={4}: {ppl_dq_bs4:.4f}")
print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
print()
ppl_dq_bs12 = measure_perplexity(model_dq, val_data, batch_size=12)
print(f"GPT-2 quantized perplexity on {dataset}/val.bin, batch_size={12}: {ppl_dq_bs12:.4f}")
print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.4f} GB")
print("\n")

# ------------------------------------------------------------------------------
# Deliverables 3 and 4
# ------------------------------------------------------------------------------
print("-" * 80 + "\nInference latency: decoding with M and speculative decoding with M, D\n" + "-" * 80)
target = 'gpt2-medium'
draft = 'gpt2'
# Load target model
print(f"Loading target model M ({target})")
M = GPT.from_pretrained(target, dict(dropout=0.0))
M.eval()
M.to(device)
if compile:
    M = torch.compile(M) # requires PyTorch 2.0 (optional)
print()
# Load draft model
print(f"Loading draft model D ({draft})")
D = GPT.from_pretrained(draft, dict(dropout=0.0))
D.eval()
D.to(device)
if compile:
    D = torch.compile(D) # requires PyTorch 2.0 (optional)

# Let the prompt be any sequence of 1024 tokens from the validation set
start_idx = torch.randint(len(val_data) - block_size, (1,)).item()
start_ids = list(val_data[start_idx : start_idx+block_size])

# Measure inference latency
def measure_inference_latency(target, draft=None, batch_size=1):
    x = torch.tensor(start_ids * batch_size, dtype=torch.long, device=device).reshape(batch_size, -1)
    with torch.no_grad():
        with ctx:
            t0 = time.time()
            if draft:
                torch.manual_seed(1337)
                y = target.generate_speculative(x, max_new_tokens, draft, temperature=temperature, top_k=top_k, num_speculative=speculative_tokens)
            else:
                torch.manual_seed(1337) 
                y = target.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            t1 = time.time()
    dt = t1 - t0
    print(f"Inference latency, batch size {batch_size}: {dt / max_new_tokens:.4f} sec/tok ({max_new_tokens} tokens, context size {block_size})")
    return y[0].tolist()

print(f"\nStandard decoding with M ({target})")
generations_M = measure_inference_latency(M)
print(f"\nSpeculative decoding with M ({target}), D ({draft})")
generations_spec = measure_inference_latency(M, D)

# We usually have num_matching = 1074 = 1024 + 50 when M = gpt2-large, D = gpt2
# We usually have num_matching = 1032 = 1024 + 50 when M = gpt2-medium, D = gpt2
num_matching = len([1 for i in range(len(generations_M)) if generations_M[i] == generations_spec[i]])

del M
del D
gc.collect()
torch.cuda.empty_cache()
print("\n")

print("-" * 80 + "\nInference latency: decoding with M and speculative decoding with M, M quantized\n" + "-" * 80)
target = 'gpt2'
draft = 'gpt2'
# Load target model
print(f"Loading target model M ({target})")
M = GPT.from_pretrained(target, dict(dropout=0.0))
M.eval()
M.to(device)
if compile:
    M = torch.compile(M) # requires PyTorch 2.0 (optional)
print()
# Load draft model
print(f"Loading draft model D ({draft})")
D = GPT_Q.from_pretrained(draft, dict(dropout=0.0))
print(f"Quantizing draft model D ({draft})")
D.quantize_all_parameters()
D.eval()
D.to(device)
if compile:
    D = torch.compile(D) # requires PyTorch 2.0 (optional)

print(f"\nStandard decoding with M ({target})")
generations_M = measure_inference_latency(M)
print(f"\nSpeculative decoding with M ({target}), D ({draft} quantized)")
generations_M_quantized = measure_inference_latency(M, model_dq)

del M
del D
gc.collect()
torch.cuda.empty_cache()