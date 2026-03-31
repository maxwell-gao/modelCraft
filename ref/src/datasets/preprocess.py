import os
from tqdm import tqdm

import numpy as np

import numpy as np
import tiktoken
import datasets
from datasets import load_dataset, DatasetDict

import argparse

import sys

sys.path.append(".")
sys.path.append("../../")

from transformers import AutoTokenizer
from utils.util import setup_logger

log = setup_logger('preprocess')


def get_tokenizer(args):
    if args.pretrained_tokenizer == "gpt2":
        args.enc = tiktoken.get_encoding("gpt2")
        args.hf_tokenizer = False
        
# Configuration
NUM_SAMPLES = 8000*512  # Number of training samples to preprocess
SEQ_LENGTH = 1024     # Sequence length for each sample
num_proc = 32
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

def process(example):
    # Tokenize the dataset
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

if __name__ == "__main__":
    # Math Pre-processing with limited samples
    log.info(f"Loading open-web-math-pro dataset...")
    dataset = load_dataset("gair-prox/open-web-math-pro", split='train', streaming=False, trust_remote_code=True)
    
    log.info(f"Splitting dataset into train/validation...")
    splits = dataset.train_test_split(test_size=0.005, seed=42, shuffle=True)
    train_dataset = splits['train']
    test_dataset = splits['test']
    
    # Take only NUM_SAMPLES from training set
    log.info(f"Taking {NUM_SAMPLES} samples from training set...")
        
    split_dataset = DatasetDict()
    #split_dataset['train'] = train_dataset
    split_dataset['test'] = test_dataset

    log.info(f"Tokenizing datasets...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        num_proc=num_proc,
        desc="tokenizing the splits",
    )

    for split, dset in tokenized.items():
        log.info(f"Processing {split} split with {len(dset)} samples...")
        
        # Calculate total tokens needed (SEQ_LENGTH + 1 for non-overlapping targets)
        if split == 'train':
            target_tokens = NUM_SAMPLES * (SEQ_LENGTH + 1)
        else:
            target_tokens = len(dset) * (SEQ_LENGTH +1 ) * 100
        
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(target_tokens,))
        
        idx = 0
        total_batches = min(1024, len(dset))
        
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            
            # Only write up to target_tokens
            tokens_to_write = min(len(arr_batch), target_tokens - idx)
            if tokens_to_write <= 0:
                break
                
            # Write into mmap
            arr[idx : idx + tokens_to_write] = arr_batch[:tokens_to_write]
            idx += tokens_to_write
            
            if idx >= target_tokens:
                break
        
        # Truncate to actual written size
        if idx < target_tokens:
            log.warning(f"Only wrote {idx} tokens out of target {target_tokens} for {split}")
            arr = np.memmap(filename, dtype=dtype, mode='r+', shape=(idx,))
        
        arr.flush()
        log.info(f"Finished writing {filename} with {idx} tokens")