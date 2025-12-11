from warnings import simplefilter

import matplotlib.pyplot as plt
import torch

from model import ExponentialMovingAverage, get_rope_vit_model
from utils import Dict2Obj, plot_loss_ess
from utils_ising import ising2d_ham
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
parser.add_argument('--resume_from_ckpt', type=str, default=None)
parser.add_argument('--wandb', dest='use_wandb', action='store_true', help="Enable wandb logging")
parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help="Disable wandb logging")
parser.set_defaults(use_wandb=False)
parser.add_argument('--wandb_project', type=str, default='mdns-ising', help="wandb project name")
parser.add_argument('--wandb_run_name', type=str, default=None, help="wandb run name")
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
dir_name = Path(args.dir_name)
dir_name.mkdir(parents=True, exist_ok=True)

def reward_fn_ising(S, beta=0.28, J=1, h=0): 
    return -beta * ising2d_ham(2*S-1, J, h)

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
       'batch_size': 128, 
       'eval_every': 20, 'eval_batch_size': 32,
       'grad_clip': False, 'gradnorm_clip': 1,
       'loss_fn': 'wdce',
       'wdce_num_replicates': 8,
       'seed': None}

wandb_run = None
if args.use_wandb:
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.dir_name,
        dir=args.dir_name,
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
with open(os.path.join(dir_name, 'config.json'), 'w') as f:
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
    model, optimizer, ema, losses, ess_train, ess_eval = train(
        model, optimizer, lambda x: reward_fn_ising(x, beta=beta, J=J, h=h), 
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
    model, optimizer, ema, losses, ess_train, ess_eval = train(
            model, optimizer, lambda x: reward_fn_ising(x, beta=args.anneal_beta, J=J, h=h), 
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
    
    model, optimizer, ema, losses, ess_train, ess_eval = train(
        model, optimizer, lambda x: reward_fn_ising(x, beta=beta, J=J, h=h), 
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
