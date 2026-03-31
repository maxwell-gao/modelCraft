import os
import sys
import argparse
import json
import wandb
import random
from collections import defaultdict
import subprocess
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

import matplotlib.pyplot as plt

import time

from tqdm import tqdm

sys.path.append("..")
sys.path.append(".")

from utils.nca import (
    generate_nca_dataset,
    generate_rules_batch,
    NCA
)

from utils.util import (
    set_seed,
    get_lr_scheduler,
    save_checkpoint,
    load_checkpoint,
    load_model,
    delete_old_checkpoint,
    setup_logger,
    wandb_log
)

from utils.models import (
    create_attention_mask,
    DownstreamLlamaModel,
    create_llama_model,
    get_grad_norm,
    get_grad_norm_attention
)

from utils.training_args import (
    NCATrainingArgs,
    nca_dataclass_to_args as dataclass_to_args,
    nca_args_to_dataclass,
    create_nca_parser as create_parser
)

from utils.tokenizers import NCA_Tokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = setup_logger('nca_train')

# ===== DATALOADERS ===== #

class NCADataset(Dataset):
    def __init__(self, seq, targets,
                 max_seq_len: int = 1024,
                 min_grid: int = 1,
                 grid_len: int = 18,
                 token: bool = False):
        super().__init__()
        self.seq = seq.long()
        self.targets = targets.long()
        self.max_seq_len = max_seq_len
        self.min_grid = min_grid
        self.grid_len = grid_len
        self.token = token

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        targets = self.targets[idx]

        # mask out negative tokens
        target = torch.where(seq < 0, torch.tensor(-100, dtype=seq.dtype), seq)
        target[:(self.min_grid*(self.grid_len)), ...] = -100 # mask minimum examples

        # shift targets and seq
        seq = seq[:-1, ...]
        targets = target[1:, ...]
        
        # pad and truncate
        if(seq.shape[0] > self.max_seq_len):
            seq = seq[:self.max_seq_len, ...]
            targets = targets[:self.max_seq_len, ...]
        elif(seq.shape[0] < self.max_seq_len):
            if(self.token):
                padding = torch.full((self.max_seq_len - seq.shape[0], seq.shape[1], seq.shape[2]), -100, dtype=seq.dtype)
                seq = torch.cat([seq, padding], dim=0)
                targets = torch.cat([targets, padding], dim=0)
            else:
                padding = torch.full((self.max_seq_len - seq.shape[0], seq.shape[1]), -100, dtype=seq.dtype)
                seq = torch.cat([seq, padding], dim=0)
                targets = torch.cat([targets, padding], dim=0)

        if(self.token):
            return seq.long(), targets.long()
        else:
            return seq.float(), targets.long()

def build_dataloader(
    args: NCATrainingArgs, 
    rng: jax.random.PRNGKey,
    num_sims: int,
    rule_seeds: jnp.ndarray,
    tokenizer: NCA_Tokenizer,
    sims: jnp.ndarray= None
):
    # generate simulations
    grid_len = (args.grid // args.patch)**2 + 2
    num_examples = int(math.ceil(args.seq_len / (grid_len)))

    if(sims is None):
        sims = generate_nca_dataset(
            rng,
            num_sims=num_sims,
            grid=args.grid,
            d_state=args.num_colors,
            n_groups=1,
            identity_bias=args.identity_bias,
            temperature=args.temperature,
            num_examples=num_examples,
            dT=args.dT,
            rule_seeds=rule_seeds,
            num_rules=rule_seeds.shape[0],
            start_step = args.init_rollout_steps
        )
    else:
        assert sims.shape[0] == num_sims, "sims must have the same number of simulations as num_sims"

    seq, targets = tokenizer.encode_task(sims)

    dataset = NCADataset(
        seq, targets,
        max_seq_len=args.seq_len,
        min_grid=args.min_grid,
        grid_len=grid_len,
        token = args.token
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    return dataloader

# ===== Tokenizer ===== #

def get_tokenizer(args: NCATrainingArgs):
    return NCA_Tokenizer(args.patch, num_colors=args.num_colors)

# ===== Evaluation Loops ===== #

def val_epoch(args: NCATrainingArgs, model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    count = 0

    base_mask = create_attention_mask(args.seq_len, additive=True).to(args.device)

    with torch.no_grad():
        for batch in dataloader:
            seq, targets = batch
            seq = seq.to(args.device)
            targets = targets.to(args.device)

            B = seq.shape[0]
            attention_mask = base_mask.repeat(B, 1, 1, 1).to(args.device)

            with torch.autocast(device_type=args.device.type, enabled=args.autocast):
                logits = model(seq, attention_mask=attention_mask)

                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)

                loss = criterion(logits, targets)

            total_loss += loss.detach().cpu().item()
            count += 1

    return total_loss, count

def eval_epoch(args: NCATrainingArgs, model, dataloader, detail=False, tokenizer=None, return_predictions: bool = False):
    model.eval()

    base_mask = create_attention_mask(args.seq_len, additive=True).to(args.device)

    def diff_fn(grid1, grid2):
        return jnp.mean((grid1==grid2))


    if(args.token and tokenizer is None):
        tokenizer = NCA_Tokenizer(args.patch, args.num_colors)

    grid_len = (args.grid // args.patch)**2 + 2

    idx = list(range(args.eval_min_grids*grid_len, args.seq_len - grid_len, grid_len))
    total_model_pred = {i // grid_len:[] for i in idx}
    total_model_targ = {i // grid_len:[] for i in idx}

    # accuracy based evaluation

    with torch.no_grad():
        for batch_seq, batch_targets in dataloader:

            if(args.token):
                batches = batch_seq.unsqueeze(0).repeat(len(idx), 1, 1)
            else:
                batches = batch_seq.unsqueeze(0).repeat(len(idx), 1, 1, 1)

            tokens_per_step = {i:[] for i in idx}

            for i in range(len(idx)):
                for l in range(grid_len-2): # don't predict last tokens
                    current_batch = batches[i].to(args.device)
                    attention_mask = create_attention_mask(current_batch.size(1)).repeat(current_batch.size(0), 1, 1 ,1).to(args.device)

                    with torch.autocast(device_type=args.device.type, enabled=args.autocast):
                        logits = model(current_batch, attention_mask=attention_mask)

                    pred_tokens = torch.argmax(F.softmax(logits[:, idx[i]+l, ...], dim=-1), dim=-1) # (B, P^2)
                    tokens_per_step[idx[i]].append(pred_tokens)

                    batches[i][:, idx[i]+l+1, ...] = pred_tokens

                predicted_tokens = torch.stack(tokens_per_step[idx[i]]) # (G, B, P^2)
                predicted_tokens = predicted_tokens.transpose(1,0) # (B, G, P^2)
                target_tokens = batch_targets[:, idx[i]:idx[i]+grid_len-2] # retrieve expected grid

                total_model_pred[idx[i] // grid_len].append(predicted_tokens)
                total_model_targ[idx[i] // grid_len].append(target_tokens)

    # compute grid prediction accuracy
    grid_delta = {i // grid_len:[] for i in idx}
    grid_input = {i // grid_len:[] for i in idx}
    grid_target = {i // grid_len:[] for i in idx}

    for i in idx:
        for pred, target in zip(total_model_pred[i // grid_len], total_model_targ[i // grid_len]):
            predicted_grid = tokenizer.decode_task(pred.detach().cpu(), dims=(args.grid, args.grid)).squeeze()
            target_grid = tokenizer.decode_task(target.detach().cpu(), dims=(args.grid, args.grid)).squeeze()
            difference = jax.vmap(diff_fn, in_axes=(0,0))(jnp.array(predicted_grid), jnp.array(target_grid))
            grid_delta[i // grid_len].append(difference)
            grid_input[i // grid_len].append(predicted_grid)
            grid_target[i // grid_len].append(target_grid)

    if detail:
        acc = {
            'full-grid': {},
            'cell-wise': {}
        }

        for i in idx:
            acc['full-grid'][i // grid_len + 1] = float(jnp.mean(jnp.concat(grid_delta[i // grid_len])==1))
            acc['cell-wise'][i // grid_len + 1] = float(jnp.mean(jnp.concat(grid_delta[i // grid_len])))

        return acc, grid_input, grid_target
    else:
        acc = []
        for k in grid_delta.keys():
            acc += grid_delta[k]
        correct_grid_per = jnp.mean(jnp.concat(acc)==1)
        return float(correct_grid_per)

def build_model(args: NCATrainingArgs):
    if args.model_type == 'llama':
        model = create_llama_model(
            vocab_size=args.input_vocab_size, # +2 for start and end tokens
            seq_length=args.seq_len,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            output_vocab=args.vocab_size
        )

        if(not args.token):
            model = DownstreamLlamaModel(
                model=model,
                input_dim=args.patch**2,
                output_dim=args.patch**2,
                num_classes=args.num_colors,
                frozen_modules=None,
                reinit_modules=None,
                input_bias=args.input_bias
            )
        return model
    else:
        raise ValueError(f"Model type {args.model_type} not supported")

# ===== Main Loop ===== #

def init_args(args: NCATrainingArgs) -> NCATrainingArgs:
    """Initialize arguments with runtime setup"""
    args.save_dir = os.path.join(args.save_dir)
    args.set_runtime_paths()
    args.input_vocab_size = args.input_vocab_size if args.input_vocab_size else args.vocab_size

    if os.path.exists(args.metrics_path) and not args.wandb_resume_run_id and args.wandb_enable and args.resume:
        with open(args.metrics_path, 'r') as f:
            metrics = json.load(f)
        args.wandb_resume_run_id = metrics['wandb_run_id']
        if args.load_dir is None:
            args.load_dir = args.save_dir
    else:
        args.resume = False

    return args

def main(args: NCATrainingArgs):
    # setup seed
    set_seed(args.seed)
    args = init_args(args)

    # setup device
    args.device = torch.device(args.device)

    log.info(f"Starting NCA training with seed: {args.seed}")
    log.info(f"Device: {args.device}")
    log.info(f"Save directory: {args.save_dir}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # initialize wandb
    if args.wandb_enable:
        if args.wandb_resume_run_id:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                resume=True,
                id=args.wandb_resume_run_id
            )
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=dataclass_to_args(args),
                resume=False
            )

    # setup randomness
    base_seed = jax.random.PRNGKey(args.seed)
    loader_rng = jax.random.split(base_seed, 3)

    tokenizer = get_tokenizer(args)

    if(args.filter_rules):
        task_rng = generate_rules_batch(
            seed = loader_rng[1],
            num_rules = args.train_num_rules + args.val_num_rules,
            tokenizer = tokenizer,
            dT = args.dT,
            n_steps = 10,
            threshold = args.filter_rules_threshold,
            upper_bound = args.filter_rules_upper_bound,
            mode = args.filter_rules_mode,
            start_step = args.init_rollout_steps,
            grid = args.grid,
            d_state = args.num_colors,
            identity_bias = args.identity_bias,
            temperature = args.temperature
        )
        log.info(f"Generated {task_rng.shape[0]} rules with threshold {args.filter_rules_threshold:.2f} and mode {args.filter_rules_mode}")
    else:
        task_rng = jax.random.split(loader_rng[1], args.train_num_rules + args.val_num_rules)
    
    train_tasks = task_rng[:args.train_num_rules]
    val_tasks = task_rng[args.train_num_rules:]

    # setup dataloader
    train_dataloader = build_dataloader(args,
                                        rng = loader_rng[0],
                                        num_sims = args.train_num_sim*args.batch_size*args.grad_accumulation_steps,
                                        rule_seeds = train_tasks,
                                        tokenizer = tokenizer)
    val_dataloader = build_dataloader(args,
                                      rng = loader_rng[1],
                                      num_sims = args.val_num_sim*args.batch_size*args.grad_accumulation_steps,
                                      rule_seeds = val_tasks,
                                      tokenizer = tokenizer)
    
    if(args.eval_enable):
        id_eval_dataloader = build_dataloader(args,
                                              rng = loader_rng[2],
                                              num_sims = args.eval_num_sim,
                                              rule_seeds = train_tasks[:args.eval_num_rules],
                                              tokenizer = tokenizer)
        ood_eval_dataloader = build_dataloader(args,
                                               rng = loader_rng[2],
                                               num_sims = args.eval_num_sim,
                                               rule_seeds = val_tasks[:args.eval_num_rules],
                                               tokenizer = tokenizer)

    # setup criterion
    criterion = CrossEntropyLoss()

    # setup the model
    model = build_model(args).to(args.device)

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"{model}")

    log.info(f"Training on {len(train_dataloader)} examples")
    log.info(f"Validating on {len(val_dataloader)} examples")

    if(args.eval_enable):
        log.info(f"Evaluating on {len(id_eval_dataloader)} examples")
        log.info(f"Evaluating on {len(ood_eval_dataloader)} examples")

    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=len(train_dataloader) * args.warmup // args.grad_accumulation_steps,
        total_steps=len(train_dataloader) * args.epochs // args.grad_accumulation_steps
    )

    # set up training loop
    metrics = defaultdict(list)
    best_val_loss = float('inf')
    best_val_loss_el = float('inf')
    start_epoch = 0
    total_iterations = 0
    
    if args.resume:
        model, optimizer, scheduler, count, start_epoch, best_val_loss, best_val_loss_el, metrics, total_iterations, _ = load_checkpoint(
            model, optimizer, scheduler, args.save_dir, args.load_dir, args.device, total_iterations=True)
        
        # Update wandb run ID in metrics if wandb is enabled
    if args.wandb_enable and 'wandb_run_id' not in metrics:
        metrics['wandb_run_id'] = wandb_run.id
    
    # compile model
    if(args.device.type != "mps" and torch.__version__ >= "2.0" and args.compile):
        compiled_model = torch.compile(model)
        torch._inductor.config.debug = False          # suppress debug prints
        torch._inductor.config.verbose_progress = False
        torch._inductor.config.triton.unique_kernel_names = True  # stop verbose duplicate warnings
    else:
        compiled_model = model

    compiled_model = compiled_model.to(args.device)
    if torch.cuda.device_count() > 1:
        compiled_model = DP(compiled_model)

    # training loop
    current_loss = 0.0
    count = 0
    epoch_count = 0
    epoch_loss = 0.0
    generate_seeds = jax.random.split(base_seed, args.epochs)
    accumulated_loss = 0.0
    accumulation_step = 0

    log.info(f"Save directory: {args.save_dir}")

    for epoch in range(start_epoch, args.epochs):
        log.info(f"=== epoch: {epoch+1} / {args.epochs} ===")
        log.info(f"training on {len(train_dataloader)} batches")

        if(args.generate_train):
            if(epoch+1 % args.generate_rules == 0):
                train_tasks = generate_rules_batch(
                    seed = generate_seeds[epoch],
                    num_rules = args.train_num_rules,
                    tokenizer = tokenizer,
                    dT = args.dT,
                    n_steps = 10,
                    threshold = args.filter_rules_threshold,
                    upper_bound = args.filter_rules_upper_bound,
                    mode = args.filter_rules_mode,
                    start_step = args.init_rollout_steps,
                    grid = args.grid,
                    d_state = args.num_colors,
                    identity_bias = args.identity_bias,
                    temperature = args.temperature,
                )

            train_dataloader = build_dataloader(args, 
                                                rng = generate_seeds[epoch],
                                                num_sims = args.train_num_sim*args.batch_size*args.grad_accumulation_steps,
                                                rule_seeds = train_tasks,
                                                tokenizer = tokenizer)

        compiled_model.train()

        if(args.autocast):
            scaler = torch.amp.GradScaler(args.device.type)

        base_mask = create_attention_mask(args.seq_len, additive=True).to(args.device) # pre-compute mask

        for batch in train_dataloader:
            seq, targets = batch
            seq = seq.to(args.device)
            targets = targets.to(args.device)

            B = seq.shape[0]
            attention_mask = base_mask.repeat(B, 1, 1, 1).to(args.device)

            if(args.mask_prob > 0.0):
                rand = torch.randn(attention_mask.shape, device=args.device)
                rand[..., 0] = -torch.finfo(rand.dtype).max
                num_mask = min(int(args.seq_len * args.mask_prob), args.seq_len-1)  # 10% per row along W
                indices = rand.topk(num_mask, dim=-1).indices  # (B, C, H, num_mask)
                keep_mask = ~torch.zeros_like(attention_mask, dtype=torch.bool).scatter(-1, indices, True).to(args.device)
                attention_mask = torch.logical_and(attention_mask, keep_mask)

            with torch.autocast(device_type=args.device.type, enabled=args.autocast):
                logits = compiled_model(seq, attention_mask=attention_mask)

                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)

                loss = criterion(logits, targets)

            # accumulate the loss
            accumulated_loss += loss.detach().cpu().item()
            accumulation_step += 1

            if args.autocast:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if accumulation_step % args.grad_accumulation_steps == 0:
                if args.autocast:
                    if args.grad_clip_enable or args.log_grad:
                        scaler.unscale_(optimizer)
                    
                    if args.grad_clip_enable:
                        clip_grad_norm_(model.parameters(), args.grad_clip)
                    
                    if args.log_grad and total_iterations % args.log_grad_freq == 0:
                        grad_norm = torch.norm(
                            torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
                        )
                        wandb_log({'iteration': total_iterations, 'train/grad_norm': grad_norm.detach().cpu().item(), "lr": scheduler.get_last_lr()[0]}, args)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip_enable:
                        clip_grad_norm_(model.parameters(), args.grad_clip)
                    
                    if args.log_grad and total_iterations % args.log_grad_freq == 0:
                        grad_norm = torch.norm(
                            torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
                        )
                        wandb_log({'iteration': total_iterations, 'train/grad_norm': grad_norm.detach().cpu().item(), "lr": scheduler.get_last_lr()[0]}, args)
                    
                    optimizer.step()
                
                # log accumulated loss
                avg_accumulated_loss = accumulated_loss / args.grad_accumulation_steps
                wandb_log({'iteration': total_iterations, 'train/loss': avg_accumulated_loss}, args)
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                epoch_loss += avg_accumulated_loss
                total_iterations += 1

                # reset accumulated
                accumulated_loss = 0.0 
                current_loss += avg_accumulated_loss
                count += 1
                epoch_count += 1         

            if(accumulation_step % args.grad_accumulation_steps == 0 and total_iterations % (args.val_freq) == 0):
                val_loss, val_count = val_epoch(args, compiled_model, val_dataloader, criterion)
                
                val_loss = val_loss / val_count
                train_loss = current_loss / count

                metrics['train/loss'].append({'loss': val_loss})
                metrics['val/loss'].append({'loss': val_loss})

                if(args.eval_enable and total_iterations % (args.val_freq * args.eval_freq) == 0):
                    id_acc = eval_epoch(args, compiled_model, id_eval_dataloader, detail=False, tokenizer=tokenizer)
                    ood_acc = eval_epoch(args, compiled_model, ood_eval_dataloader, detail=False, tokenizer=tokenizer)

                    metrics['id/acc'].append(id_acc)
                    metrics['ood/acc'].append(ood_acc)

                    log.info(f"[Iter {total_iterations:,}] ID acc: {id_acc:.4f} | OOD acc: {ood_acc:.4f}")

                    wandb_log({'iteration': total_iterations, 'train/acc': id_acc, 'ood/acc': ood_acc}, args)
                    
                log.info(f"[Iter {total_iterations:,}] Train loss: {train_loss:.8f} | Val loss: {val_loss:.8f}")

                wandb_log({'iteration': total_iterations, 'train/loss_mean': train_loss, 'ood/loss': val_loss}, args)

                if(val_loss < best_val_loss):
                    best_val_loss = val_loss
                    best_val_loss_el = val_loss

                    save_checkpoint(epoch+1, count, model, optimizer, scheduler,
                                    best_val_loss, best_val_loss_el, metrics, args.save_dir, best=True, total_iterations=total_iterations)
                    
                    delete_old_checkpoint(args.save_dir)
                    
                current_loss = 0.0
                count = 0

                compiled_model.train()

        epoch_loss = epoch_loss / epoch_count
        log.info(f"[Epoch {epoch+1}] train loss: {epoch_loss}")

        if(args.interval_save and (epoch+1) in args.intervals):
            save_checkpoint(epoch+1, count, model, optimizer, scheduler,
                            best_val_loss, best_val_loss_el, metrics, args.interval_save_path, total_iterations=total_iterations)
        
        save_checkpoint(epoch+1, count, model, optimizer, scheduler,
                            best_val_loss, best_val_loss_el, metrics, args.save_dir, total_iterations=total_iterations)
        delete_old_checkpoint(args.save_dir)

        if os.path.exists(args.metrics_path):
            os.remove(args.metrics_path)

        os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)

        with open(args.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Generate new training data if generate_train is enabled

        epoch_loss = 0.0
        epoch_count = 0

def eval_main(args: NCATrainingArgs):
    # model evaluation and visualizations
    set_seed(args.seed)
    args = init_args(args)

    # setup device
    args.device = torch.device(args.device)

    log.info(f"Starting NCA evaluation with seed: {args.seed}")
    log.info(f"Device: {args.device}")
    log.info(f"Save directory: {args.save_dir}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # set up dataloader
    base_seed = jax.random.PRNGKey(args.seed)

    # determine rule seeds
    tokenizer = get_tokenizer(args)

    if(args.filter_rules):
        task_rng = generate_rules_batch(
            seed = base_seed,
            num_rules = args.eval_num_rules,
            tokenizer = tokenizer,
            dT = args.dT,
            n_steps = 10,
            threshold = args.filter_rules_threshold,
            mode = args.filter_rules_mode,
            start_step = args.init_rollout_steps,
            grid = args.grid,
            d_state = args.num_colors,
            identity_bias = args.identity_bias,
            temperature = args.temperature
        )
        log.info(f"Generated {task_rng.shape[0]} rules with threshold {args.filter_rules_threshold:.2f} and mode {args.filter_rules_mode}")
    else:
        task_rng = jax.random.split(base_seed, args.eval_num_rules)

    eval_dataloader = build_dataloader(args,
                                       rng = base_seed,
                                       num_sims = args.eval_num_sim,
                                       rule_seeds = task_rng,
                                       tokenizer = tokenizer)

    # set up model
    model = build_model(args).to(args.device)

    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"{model}")

    # load model
    model = load_model(model, args.save_dir, args.load_dir)

    # set up criterion
    criterion = CrossEntropyLoss()

    # set up evaluation loop
    acc, grid_output, grid_target = eval_epoch(args, model, eval_dataloader, tokenizer=tokenizer, detail=True)
    log.info(f"Accuracies: {acc}")

    # save grid_input and grid_target
    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir)

    with open(os.path.join(args.eval_dir, 'accuracy.json'), 'w') as f:
        json.dump(acc, f)

    # plot some examples
    plot_examples(grid_output, grid_target, args.eval_icl_step, example_idx=list(range(args.eval_num_examples)), save_dir=args.eval_dir)

def plot_examples(grid_output, grid_target, icl_step, example_idx=[0], save_dir=None):
    cmap = plt.get_cmap('tab20', lut=20)   # lut fixes number of discrete colors
    # plot some examples 
    icl = [int(k) for k in grid_output.keys()]
    examples = list(range(min(icl), max(icl)+1, icl_step))

    n_col = len(examples)
    n_row = len(example_idx)*2

    fig, axs = plt.subplots(n_row, n_col, figsize=(n_col*2, n_row*2))

    for i in range(n_row // 2):
        for j in range(n_col):
            axs[2*i, j].imshow(grid_output[examples[j]][0][example_idx[i]], cmap=cmap)
            axs[2*i, j].set_title(f"Output ICL: {examples[j]}")
            axs[2*i, j].axis('off')

            axs[2*i+1, j].imshow(grid_target[examples[j]][0][example_idx[i]], cmap=cmap)
            axs[2*i+1, j].set_title(f"Target ICL: {examples[j]}")
            axs[2*i+1, j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'examples.png'))
    
# ===== Main ===== #
if __name__ == "__main__":
    parser = create_parser()
    parsed_args = parser.parse_args()
    training_args = nca_args_to_dataclass(parsed_args)

    if training_args.eval_mode:
        eval_main(training_args)
    else:
        main(training_args)