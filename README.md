# Conditional WGAN-GP with DeepSpeed on MNIST

This project implements a Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) trained on the MNIST dataset. It uses DeepSpeed to enable memory-efficient training and mixed-precision computation. The GAN is conditioned on digit labels and built using PyTorch.

---

## Features

- Conditional generation using label embeddings
- WGAN-GP loss for stable training
- DeepSpeed integration (FP16 + ZeRO optimization)
- Generator and Critic training loops with gradient penalty
- Sample image visualization every few epochs

---

## Requirements

```bash
pip install torch torchvision matplotlib deepspeed
