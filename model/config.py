from dataclasses import dataclass
from pydantic.dataclasses import dataclass
from pydantic import BaseModel, Field
from diffusers.models.embeddings import get_timestep_embedding, Timesteps

class train_config(BaseModel):
    train_batch_size: int = Field(gt = 3, lt = 1025)
    num_epochs: int = Field(gt = 0, lt = 5000)
    lr_warmup_steps: int
    save_model_epochs: int
    num_train_timesteps: int
    output_dir: str = "./"
    print_epoch_loss: int
    learning_rate: float = Field(default=1e-4, gt=0)
    seed: int = 42
    max_minutes: float = Field(default=25, ge=0)

config2 = train_config(train_batch_size=32, num_epochs=2000, lr_warmup_steps=50, save_model_epochs=200, num_train_timesteps=1000, print_epoch_loss=50)

# keep track on timestamps for step by step noising process to input in model
time_embed_dim = 64
time_proj = Timesteps(
                time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0
            )


def training_options(default_lr=1e-4):
    import argparse
    parser = argparse.ArgumentParser(description="Run the existing diffusion training loop")
    parser.add_argument("--data", default="datasets/grasps.npz")
    parser.add_argument("--output", default="checkpoints/grasp")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-minutes", type=float, default=25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    config = train_config(train_batch_size=args.batch_size, num_epochs=args.epochs,
                         lr_warmup_steps=50, save_model_epochs=25, num_train_timesteps=args.steps,
                         output_dir=args.output, print_epoch_loss=1, learning_rate=args.lr,
                         seed=args.seed, max_minutes=args.max_minutes)
    return args, config


def prepare_data(cand_list, config, sequence_length=4):
    import random
    import numpy as np
    import torch
    from data.processing import train_dataset, fit_normalizer
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if len(cand_list) < 2:
        raise ValueError("Collect at least two scenes for validation")
    indices = np.random.default_rng(config.seed).permutation(len(cand_list))
    split = max(1, min(len(cand_list) - 1, round(.15 * len(cand_list))))
    training = [cand_list[i] for i in indices[split:]]
    validation = [cand_list[i] for i in indices[:split]]
    normalizer = fit_normalizer(training)
    metadata = {"train_scene_indices": indices[split:].tolist(), "validation_scene_indices": indices[:split].tolist()}
    return (train_dataset(training, normalizer, sequence_length),
            train_dataset(validation, normalizer, sequence_length), normalizer, metadata)


def validation_loss(model, loader, scheduler, device, kind, seed):
    import torch
    import torch.nn.functional as F
    from model.unet1d import hugging_input
    generator = torch.Generator().manual_seed(seed + 10000)
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for grasp, block in loader:
            grasp, block = grasp.to(device), block.to(device)
            noise = torch.randn(grasp.shape, generator=generator).to(device)
            noise[..., 1:] = 0
            timesteps = torch.randint(scheduler.config.num_train_timesteps, (len(grasp),), generator=generator).to(device)
            noisy = scheduler.add_noise(grasp, noise, timesteps)
            predicted = model(noisy, block, timesteps) if kind == "custom" else model(hugging_input(noisy, block, timesteps), timestep=timesteps).sample
            target = grasp if scheduler.config.prediction_type == "sample" else noise
            total += F.mse_loss(predicted[..., 0], target[..., 0]).item() * len(grasp)
            count += len(grasp)
    model.train()
    import math
    result = total / count
    if not math.isfinite(result):
        raise RuntimeError("Non-finite validation loss; inspect the data and learning rate")
    return result


def checkpoint_info(config, normalizer, split, kind, sequence_length, model):
    return {"format_version": 2, "config": config.model_dump(), "normalizer": normalizer,
            "split": split, "model_kind": kind, "sequence_length": sequence_length,
            "prediction_type": "sample" if kind == "custom" else "epsilon",
            "model_config": dict(model.config) if kind == "hugging" else None,
            "parameter_count": sum(p.numel() for p in model.parameters())}


def save_progress(output_dir, state, history, best):
    import csv
    import os
    import torch
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state, os.path.join(output_dir, "last.pt"))
    if state["validation_loss"] < best:
        best = state["validation_loss"]
        torch.save(state, os.path.join(output_dir, "best.pt"))
    with open(os.path.join(output_dir, "history.csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return best
