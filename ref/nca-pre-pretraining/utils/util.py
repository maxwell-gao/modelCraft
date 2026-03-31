import numpy as np
import math
import random
import logging

import os
import json

from collections import defaultdict

import torch
from torch.optim.lr_scheduler import LambdaLR
import wandb

#############################################################
#  JSONL Logger Setup
#############################################################

def write_jsonl(path, iterator):
    with open(path, "w", encoding="utf-8") as f:
        for obj in iterator:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

#############################################################
#  Logger Setup
#############################################################

def setup_logger(name='training', level=logging.INFO):
    """
    Set up logger for training scripts
    
    Args:
        name (str): Name of the logger
        level: Logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger('util')

#############################################################
#  General utils
#############################################################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def get_lr_scheduler(optimizer, warmup_steps, total_steps, decrease_mode='cosin'):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            if decrease_mode == 'cosin':
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            elif decrease_mode == 'linear':
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 1.0 - progress
            elif decrease_mode == 'const':
                return 1.0
            else:
                raise ValueError(f"Invalid decrease mode: {decrease_mode}")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)

#############################################################
#  Model utilities
#############################################################

def sort_checkpoint(name):
    """Sort checkpoints by iteration number"""
    if 'iteration' in name:
        # New format: checkpoint_iteration_XXXX.pth
        return int(name.replace('.pth', '').split('_')[-1])
    else:
        # Legacy format: gpt_model_XXXX.pth (epoch-based)
        return int(name.replace('.pth', '').split('_')[-1])

def save_checkpoint(epoch, count, model, optimizer, scheduler, best_val_loss, best_val_loss_el, metrics, save_dir, best=False, total_iterations=0, by_epoch=True):
    if(hasattr(model, 'module')):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    ckpt = {
        'model': model_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'count': count,
        'best_val_loss': best_val_loss,
        'best_val_loss_el': best_val_loss_el,
        'metrics': metrics,
        'total_iterations': total_iterations
    }

    if(by_epoch):
        prefix = epoch
    else:
        prefix = f'{total_iterations}_iter'

    if(best):
        torch.save(ckpt, os.path.join(save_dir, f'best_model_{prefix}.pth'))
    else:
        torch.save(ckpt, os.path.join(save_dir, f'model_{prefix}.pth'))


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def load_checkpoint(model, optimizer, scheduler, save_dir, load_dir, device=None, total_iterations=False):
    """
    Load checkpoint with enhanced support for iteration-based checkpoints and dataloader state
    
    Returns:
        If total_iterations=True: model, optimizer, scheduler, count, epoch, best_val_loss, best_val_loss_el, metrics, loaded_total_iterations, dataloader_state
        If total_iterations=False: model, optimizer, scheduler, count, epoch, best_val_loss, best_val_loss_el, metrics
    """
    
    if load_dir and os.path.isfile(os.path.join(save_dir, load_dir)):
        model_f = os.path.join(save_dir, load_dir)
    elif load_dir and os.path.isdir(load_dir):
        # Search for last checkpoint in load_dir (prefer iteration-based, fallback to epoch-based)
        iteration_checkpoints = [f for f in os.listdir(load_dir) if f.startswith('checkpoint_iteration_')]
        epoch_checkpoints = [f for f in os.listdir(load_dir) if f.startswith('model')]
        
        if len(iteration_checkpoints) > 0:
            # Use iteration-based checkpoints (newest format)
            iteration_checkpoints.sort(key=lambda x: sort_checkpoint(x))
            model_f = os.path.join(load_dir, iteration_checkpoints[-1])
        elif len(epoch_checkpoints) > 0:
            # Fallback to epoch-based checkpoints (legacy format)
            epoch_checkpoints.sort(key=lambda x: sort_checkpoint(x))
            model_f = os.path.join(load_dir, epoch_checkpoints[-1])
        else:
            print(f'No checkpoints found in {load_dir}')
            if total_iterations:
                return model, optimizer, scheduler, 0, 0, float('inf'), float('inf'), defaultdict(list), 0, None
            else:
                return model, optimizer, scheduler, 0, 0, float('inf'), float('inf'), defaultdict(list)
    else:
        # Search for last checkpoint in save_dir if no model specified
        iteration_checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_iteration_')]
        epoch_checkpoints = [f for f in os.listdir(save_dir) if f.startswith('model')]
        
        if len(iteration_checkpoints) > 0:
            # Use iteration-based checkpoints (newest format)
            iteration_checkpoints.sort(key=lambda x: sort_checkpoint(x))
            model_f = os.path.join(save_dir, iteration_checkpoints[-1])
        elif len(epoch_checkpoints) > 0:
            # Fallback to epoch-based checkpoints (legacy format)
            epoch_checkpoints.sort(key=lambda x: sort_checkpoint(x))
            model_f = os.path.join(save_dir, epoch_checkpoints[-1])
        else:
            print(f'No checkpoints found in {save_dir}')
            if total_iterations:
                return model, optimizer, scheduler, 0, 0, float('inf'), float('inf'), defaultdict(list), 0, None
            else:
                return model, optimizer, scheduler, 0, 0, float('inf'), float('inf'), defaultdict(list)
    
    print(f"Loading checkpoint from: {model_f}")
    sd = torch.load(model_f, weights_only=False, map_location=device)

    # Load model state with error handling for size mismatches
    try:
        model.load_state_dict(sd['model'])
    except:
        if 'input_proj.weight' in sd['model'] and 'output_proj.weight' in sd['model']:
            del sd['model']['input_proj.weight']
            del sd['model']['output_proj.weight']
            missing_keys, unexpected_keys = model.load_state_dict(sd['model'], strict=False)
            if missing_keys:
                print(f"Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading model: {unexpected_keys}")
        else:
            raise RuntimeError('Error loading model')
    
    # Load optimizer state
    if optimizer is not None:
        try:
            optimizer.load_state_dict(sd['optimizer'])
        except:
            print(f'No optimizer found in {model_f}')
    
    # Load scheduler state
    if scheduler is not None:
        try:
            scheduler.load_state_dict(sd['scheduler'])
        except:
            print(f'No scheduler found in {model_f}')
    
    count = sd.get('count', 0)
    epoch = sd.get('epoch', 0)
    best_val_loss = sd.get('best_val_loss', float('inf'))
    best_val_loss_el = sd.get('best_val_loss_el', float('inf'))
    metrics = sd.get('metrics', defaultdict(list))
    
    # Load total_iterations and dataloader_state if requested and available
    loaded_total_iterations = sd.get('total_iterations', 0) or sd.get('current_iterations', 0) or sd.get('count', 0)
    dataloader_state = sd.get('dataloader_state', None)

    # Move optimizer to device
    if optimizer is not None:
        try:
            move_optimizer_to_device(optimizer, device)
        except:
            print('Not moving optimizer to device b/c not loaded')
    
    if total_iterations:
        return model, optimizer, scheduler, count, epoch, best_val_loss, best_val_loss_el, metrics, loaded_total_iterations, dataloader_state
    else:
        return model, optimizer, scheduler, count, epoch, best_val_loss, best_val_loss_el, metrics


def delete_old_checkpoint(save_dir, save_total_limit=2):
    for prefix in ['model', 'best_model', 'checkpoint_iteration']:
        checkpoints = [f for f in os.listdir(save_dir) if f.startswith(prefix)]
        checkpoints.sort(key=lambda x: sort_checkpoint(x))
        if len(checkpoints) > save_total_limit:
            oldest_f = os.path.join(save_dir, checkpoints[0])
            os.remove(oldest_f)

def load_model(model, save_dir, load_dir, pretrain_pos=True, strict=False):
    if(load_dir):
        model_f = os.path.join(save_dir, load_dir)
    else:
        # search for last checkpoint if no model specified
        print(save_dir)
        checkpoints = [f for f in os.listdir(save_dir) if 'model' in f]
        assert(len(checkpoints) > 0)
        checkpoints.sort(key=lambda x: sort_checkpoint(x))
        model_f = os.path.join(save_dir, checkpoints[-1])
    
    logger.info(f"Loading model from: {model_f}")
    sd = torch.load(model_f, weights_only=False, map_location='cpu')
    model_param = sd['model']

    if(not(pretrain_pos)):
        pos_layer = 'wpe'
        model_param = {k:v for k,v in model_param.items()
                       if not(k.startswith(pos_layer))}

    model.load_state_dict(model_param, strict=strict)

    return model

def log_model_parameters(model):
    """
    Log the total number of parameters and trainable parameters of a model.
    
    Args:
        model: PyTorch model
        logger: Logger instance to use for logging. If None, uses the default logger.
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model Parameters Summary:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info(f"  Trainable percentage: {(trainable_params / total_params * 100):.2f}%")
    
    return total_params, trainable_params

def wandb_log(log, args):
    if(args.wandb_enable):
        wandb.log(log)