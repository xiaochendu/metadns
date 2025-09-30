# MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2508.10684)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![X](https://img.shields.io/badge/X-000000?logo=x&logoColor=white&style=flat-square)](https://x.com/YuchenZhu_ZYC/status/1963269587413168524)

Welcome to the official implementation for **MDNS** (accepted at NeurIPS 2025), for  training <span style="color:#A020F0">**neural samplers**</span> for discrete distribution with <span style="color:#A020F0">**masked discrete diffusion models.**</span>

![Demonstration](assets/samples.png)
![method](assets/third_image.png)

## Environment
```
conda env create -f environment.yml
conda activate MDNS
```

## Training
The commands for training MDNS on Ising and Potts models are listed in `train_commands.sh`

## Checkpoints
We included checkpoints trained with MDNS for Ising and Potts model on 2D square lattice under the folder `checkpoints`. 

- For Ising model, we include those on 16x16 and 24x24 square lattice, across three different temperatures $\beta_{\text{high}} = 0.28$, $\beta_{\text{critical}} = 0.4407$, and $\beta_{\text{low}} = 0.6$. We named them correspondingly as `ising_high.pth`, `ising_crit.pth` and `ising_low.pth` under the directory.
- For Potts model, we include those on 16x16 square lattice, across three different temperatures $\beta_{\text{high}} = 0.4$, $\beta_{\text{critical}} = 1.005$, and $\beta_{\text{low}} = 1.2$. We named them correspondingly as `potts_high.pth`, `potts_crit.pth` and `potts_low.pth` under the directory.


## Evaluation
We include evaluation and visualization script of Ising and Potts model in `ising_model_eval.ipynb` and `potts_model_eval.ipynb` respectively. 


## Citation
If you find our work and repo help, we would appreciate your citations :smiling_face_with_three_hearts:


```
@article{zhu2025mdns,
  title={MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control},
  author={Zhu, Yuchen and Guo, Wei and Choi, Jaemoo and Liu, Guan-Horng and Chen, Yongxin and Tao, Molei},
  journal={arXiv preprint arXiv:2508.10684},
  year={2025}
}
```




