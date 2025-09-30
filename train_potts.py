import torch
from model import get_rope_vit_model, ExponentialMovingAverage
from utils import Dict2Obj, plot_loss_ess
from utils_train import train
from utils_potts import potts2d_ham
from warnings import simplefilter
import matplotlib.pyplot as plt
simplefilter(action='ignore', category=FutureWarning)
import os
import argparse
from pprint import pformat
import json

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--L', type=int, default=16)
parser.add_argument('--q', type=int, default=3)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--J', type=float, default=1)
parser.add_argument('--dir_name', type=str, default=None)
parser.add_argument('--num_epochs', type=int, default=50000)
parser.add_argument('--use_anneal', action='store_true')
parser.add_argument('--anneal_beta', type=float, default=None)
parser.add_argument('--anneal_epochs', type=int, default=None)
parser.add_argument('--resume_from_ckpt', type=str, default=None)
args = parser.parse_args()

# Critical temp = log(1 + sqrt(q))

if args.use_anneal:
    assert args.anneal_beta is not None, "anneal_beta must be specified if anneal is True"
    assert args.anneal_epochs is not None, "anneal_epochs must be specified if anneal is True"
    
device = args.device
L = args.L
D=L**2; 
q = args.q
N = q + 1; 
beta= args.beta
J = args.J
h = 0
resume_path = args.resume_from_ckpt
anneal_beta = args.anneal_beta
dir_name = f'exp_local/L_{L}_potts_q_{q}_beta_{beta}_J_{J}/{args.dir_name}'
os.makedirs(dir_name, exist_ok=True)

def reward_fn_potts(S, beta = 0.5, J = 1, q = 3):
    return -beta * potts2d_ham(S, J, q)

cfg = {'tokens':q,
       "anneal": args.use_anneal,
        "anneal_beta": anneal_beta if args.use_anneal else None,
        "anneal_epochs": args.anneal_epochs,
        "resume_from_ckpt": resume_path,
       "L": L, 
       "q": q, 
       "beta": beta, 
       "J": J, 
       "dir_name": args.dir_name,
       'model':{'name': 'small_radd','type': 'ddit_wot','hidden_size': 128,
                'cond_dim': 128,'length': D,'n_blocks': 4,'n_heads': 4, 
                'dropout': 0.0,'use_checkpoint': False,'dtype': 'bfloat16'},
       'grad_clip': False, 'gradnorm_clip':1,'num_epochs': args.num_epochs, 'resample_every_n_step':10,
       'warmup_steps':0,
       'num_accum_steps':1,'batch_size': 256,'copy_flag_temp':None,'truncate_steps':32,
       'truncate_kl':False,'total_num_steps':32,'gumbel_temp':1,'gumbel_temp_schedule':'const', 
       'gumbel_temp_min': 0.1, 'eval_every':20, 'eval_batch_size': 32,
       'loss_fn':'wdce', 'wdce_num_replicates': 8, 'seed': None}

model = get_rope_vit_model(L, vocab_size=cfg['tokens'] + 1, embed_dim=cfg['model']['hidden_size'], depth=cfg['model']['n_blocks'], num_heads=cfg['model']['n_heads'], dtype=cfg['model']['dtype'], device=device)
ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.00)

print('Model: num of params: {}, size: {:.2f} MB'.format(
    sum(p.numel() for p in model.parameters()),
    sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)))

scheduler = None


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
        model, optimizer, lambda x: reward_fn_potts(x, beta = beta, J = J, q = q), 
        Dict2Obj(cfg), device, ema=ema, scheduler=scheduler, num_epochs=args.num_epochs,
        losses=losses, ess_train=ess_train, ess_eval=ess_eval)
    
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
            model, optimizer, lambda x: reward_fn_potts(x, beta = args.anneal_beta, J = J, q = q), 
            Dict2Obj(cfg), device, num_epochs=args.anneal_epochs, 
            ema=ema, scheduler=scheduler)
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
        model, optimizer, lambda x: reward_fn_potts(x, beta = beta, J = J, q = q), 
        Dict2Obj(cfg), device, num_epochs=args.num_epochs, 
        ema=ema, scheduler=scheduler, losses=losses, ess_train=ess_train, ess_eval=ess_eval)
    fig, ax = plot_loss_ess(losses, ess_train, ess_eval=ess_eval)
    plt.savefig(f"{dir_name}/loss_ess.png")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'losses': losses, 'ess_train': ess_train, 
        'ess_eval': ess_eval,
        'cfg': cfg}, f'{dir_name}/weights_final.pth')
