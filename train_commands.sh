## Potts Model

## High temperature
python train_potts.py \
    --L 16 \
    --q 3 \
    --beta 0.5 \
    --J 1 \
    --device cuda:0 \
    --dir_name "potts_high" \
    --num_epochs 50000

## Critical temperature
python train_potts.py \
    --L 16 \
    --q 3 \
    --beta 1.005 \
    --anneal_beta 0.5 \
    --use_anneal \
    --anneal_epochs 30000 \
    --J 1 \
    --device cuda:0 \
    --dir_name "potts_crit" \
    --num_epochs 50000

## Low temperature
python train_potts.py \
    --L 16 \
    --q 3 \
    --beta 1.2 \
    --anneal_beta 0.5 \
    --use_anneal \
    --anneal_epochs 30000 \
    --J 1 \
    --device cuda:0 \
    --dir_name "potts_low" \
    --num_epochs 50000

## Ising Model Training


## High temperature
python train_ising.py \
    --L 16 \
    --beta 0.28 \
    --J 1 \
    --device cuda:0 \
    --dir_name "ising_high" \
    --num_epochs 50000


## Critical temperature
python train_ising.py \
    --L 16 \
    --beta 0.4407 \
    --J 1 \
    --use_anneal \
    --anneal_beta 0.28 \
    --anneal_epochs 20000 \
    --device cuda:0 \
    --dir_name "ising_crit" \
    --num_epochs 50000

## Low temperature

python train_ising.py \
    --L 16 \
    --beta 0.6 \
    --J 1 \
    --use_anneal \
    --anneal_beta 0.28 \
    --anneal_epochs 20000 \
    --device cuda:0 \
    --dir_name "ising_low" \
    --num_epochs 50000


