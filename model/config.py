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

config2 = train_config(train_batch_size=32, num_epochs=2000, lr_warmup_steps=50, save_model_epochs=200, num_train_timesteps=1000, print_epoch_loss=50)

time_embed_dim = 64
time_proj = Timesteps(
                time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0
            )
