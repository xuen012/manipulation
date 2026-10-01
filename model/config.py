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