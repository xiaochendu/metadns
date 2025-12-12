from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import ExponentialMovingAverage, get_rope_vit_model
from utils import Dict2Obj, plot_loss_ess
from utils_ising import ising2d_ham, reward_fn_ising
from utils_train import train

simplefilter(action='ignore', category=FutureWarning)
import argparse
import json
import os
from pathlib import Path
from pprint import pformat

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--L', type=int, default=24)
parser.add_argument('--beta', type=float, default=0.28)
parser.add_argument('--J', type=float, default=1)
parser.add_argument('--dir_name', type=str, default=None)
parser.add_argument('--num_epochs', type=int, default=100000)
parser.add_argument('--use_anneal', action='store_true')
parser.add_argument('--anneal_beta', type=float, default=None)
parser.add_argument('--anneal_epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--resume_from_ckpt', type=str, default=None)
parser.add_argument('--wandb', dest='use_wandb', action='store_true', help="Enable wandb logging")
parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help="Disable wandb logging")
parser.set_defaults(use_wandb=False)
parser.add_argument('--wandb_project', type=str, default='mdns-ising', help="wandb project name")
parser.add_argument('--wandb_run_name', type=str, default=None, help="wandb run name")
parser.add_argument('--temps', type=float, nargs='+', default=None,
                    help='List of temperatures (overrides --beta). If provided, batch_size must be divisible by num_temps * num_fields.')
parser.add_argument('--fields', type=float, nargs='+', default=None,
                    help='List of field values. If provided, batch_size must be divisible by num_temps * num_fields.')
parser.add_argument('--sample_delta_temp', action='store_true',
                    help='Sample temperatures dynamically around provided temps')
parser.add_argument('--sample_delta_field', action='store_true',
                    help='Sample fields dynamically around provided fields')
parser.add_argument('--delta_temp', type=float, default=0.2,
                    help='Temperature sampling range for dynamic sampling')
parser.add_argument('--delta_field', type=float, default=0.05,
                    help='Field sampling range for dynamic sampling')
parser.add_argument('--min_temp', type=float, default=1.667,
                    help='Minimum temperature for dynamic sampling')
parser.add_argument('--max_temp', type=float, default=3.5714,
                    help='Maximum temperature for dynamic sampling')
args = parser.parse_args()

if args.use_anneal:
    assert args.anneal_beta is not None, "anneal_beta must be specified if anneal is True"
    assert args.anneal_epochs is not None, "anneal_epochs must be specified if anneal is True"
    
device = args.device
L = args.L
D = L**2
beta = args.beta
J = args.J
h = 0
resume_path = args.resume_from_ckpt
anneal_beta = args.anneal_beta
# Handle None dir_name (when script is imported rather than run directly)
if args.dir_name is None:
    dir_name = Path('exp_local/default')
else:
    dir_name = Path(args.dir_name)
dir_name.mkdir(parents=True, exist_ok=True)

# Handle multiple temperatures/fields
# If temps provided, use them; otherwise use single beta converted to temperature
temps = args.temps if args.temps is not None else [1.0 / beta]  # Convert beta to temperature
fields = args.fields if args.fields is not None else [0.0]
sample_delta_temp = args.sample_delta_temp if args.sample_delta_temp is not None else False
sample_delta_field = args.sample_delta_field if args.sample_delta_field is not None else False
delta_temp = args.delta_temp if args.delta_temp is not None else 0.2
delta_field = args.delta_field if args.delta_field is not None else 0.05
min_temp = args.min_temp if args.min_temp is not None else 1.667
max_temp = args.max_temp if args.max_temp is not None else 3.5714

cfg = {'tokens': 2,
       "anneal": args.use_anneal,
       "anneal_beta": anneal_beta if args.use_anneal else None,
       "anneal_epochs": args.anneal_epochs,
       "resume_from_ckpt": resume_path,
       "L": L,
       "beta": beta,
       "J": J,
       "dir_name": args.dir_name,
       'model': {'hidden_size': 64, 'n_blocks': 4, 'n_heads': 4, 'length': D, 
                 'use_checkpoint': False, 'dtype': 'bfloat16'},
       'num_epochs': args.num_epochs,
       'resample_every_n_step': 10,
       'batch_size': args.batch_size, 
       'eval_every': 20,
       'eval_batch_size': args.eval_batch_size,
       'grad_clip': False, 'gradnorm_clip': 1,
       'loss_fn': 'wdce',
       'wdce_num_replicates': 8,
       'seed': None,
       'temps': temps,
       'fields': fields,
       'sample_delta_temp': sample_delta_temp,
       'sample_delta_field': sample_delta_field,
       'delta_temp': delta_temp,
       'delta_field': delta_field,
       'min_temp': min_temp,
       'max_temp': max_temp,
       'J': J}

# Check batch size compatibility if using multiple temps/fields
if len(temps) > 1 or len(fields) > 1:
    num_conditions = len(temps) * len(fields)
    if cfg['batch_size'] % num_conditions != 0:
        raise ValueError(
            f"batch_size ({cfg['batch_size']}) must be divisible by "
            f"num_temps ({len(temps)}) * num_fields ({len(fields)}) = {num_conditions}"
        )
    if cfg['eval_batch_size'] % num_conditions != 0:
        raise ValueError(
            f"eval_batch_size ({cfg['eval_batch_size']}) must be divisible by "
            f"num_temps ({len(temps)}) * num_fields ({len(fields)}) = {num_conditions}"
        )

wandb_run = None
if args.use_wandb:
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.dir_name,
        dir=str(dir_name),
        config=cfg,
    )

model = get_rope_vit_model(L, embed_dim=cfg['model']['hidden_size'], 
                          depth=cfg['model']['n_blocks'], 
                          num_heads=cfg['model']['n_heads'], 
                          dtype=cfg['model']['dtype'], 
                          device=device)
ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.00)

print('Model: num of params: {}, size: {:.2f} MB'.format(
    sum(p.numel() for p in model.parameters()),
    sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)))

print(f"Training config:\n{pformat(cfg)}")
with open(dir_name / 'config.json', 'w') as f:
    json.dump(cfg, f, indent=4)

if not args.use_anneal:
    if resume_path is not None:
        print("Loading checkpoint from: ", resume_path)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losses = checkpoint['losses']
        ess_train = checkpoint['ess_train']
        ess_eval = checkpoint['ess_eval']
    else:
        print("No checkpoint provided, starting from scratch")
        losses = []
        ess_train = []
        ess_eval = []
        
    model.train()
    # Create reward function wrapper that accepts optional beta/h for per-sample values
    default_beta = beta
    default_h = h
    def reward_fn(x, beta=None, h=None, J=J):
        """Reward function wrapper that handles scalar or per-sample betas/fields."""
        beta_val = beta if beta is not None else default_beta
        h_val = h if h is not None else default_h
        return reward_fn_ising(x, beta=beta_val, J=J, h=h_val)
    
    model, optimizer, ema, losses, ess_train, ess_eval = train(
        model, optimizer, reward_fn, 
        Dict2Obj(cfg), device, ema=ema, num_epochs=args.num_epochs,
        losses=losses, ess_train=ess_train, ess_eval=ess_eval,
        wandb_run=wandb_run, L=L)
    
    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{dir_name}/loss_ess.png")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'losses': losses, 'ess_train': ess_train, 
        'ess_eval': ess_eval,
        'cfg': cfg}, f'{dir_name}/weights.pth')
else:
    model.train()
    # Create reward function for annealing phase
    default_beta_anneal = args.anneal_beta
    default_h_anneal = h
    def reward_fn_anneal(x, beta=None, h=None, J=J):
        beta_val = beta if beta is not None else default_beta_anneal
        h_val = h if h is not None else default_h_anneal
        return reward_fn_ising(x, beta=beta_val, J=J, h=h_val)
    
    model, optimizer, ema, losses, ess_train, ess_eval = train(
            model, optimizer, reward_fn_anneal, 
            Dict2Obj(cfg), device, num_epochs=args.anneal_epochs, ema=ema,
            wandb_run=wandb_run, L=L)
    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{dir_name}/loss_ess_anneal.png")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'losses': losses, 'ess_train': ess_train, 
        'ess_eval': ess_eval,
        'cfg': cfg}, f'{dir_name}/weights_warmup.pth')
    
    # Create reward function for main training phase
    default_beta_main = beta
    default_h_main = h
    def reward_fn_main(x, beta=None, h=None, J=J):
        beta_val = beta if beta is not None else default_beta_main
        h_val = h if h is not None else default_h_main
        return reward_fn_ising(x, beta=beta_val, J=J, h=h_val)
    
    model, optimizer, ema, losses, ess_train, ess_eval = train(
        model, optimizer, reward_fn_main, 
        Dict2Obj(cfg), device, num_epochs=args.num_epochs, 
        ema=ema, losses=losses, ess_train=ess_train, ess_eval=ess_eval,
        wandb_run=wandb_run, L=L)
    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{dir_name}/loss_ess.png")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'losses': losses, 'ess_train': ess_train, 
        'ess_eval': ess_eval,
        'cfg': cfg}, f'{dir_name}/weights_final.pth')

if wandb_run is not None:
    wandb_run.finish()
