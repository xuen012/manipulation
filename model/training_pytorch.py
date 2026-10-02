from accelerate import Accelerator
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from diffusers.optimization import get_cosine_schedule_with_warmup
import os
from tqdm.auto import tqdm
from diffusers import UNet1DModel, DDPMScheduler, DDIMScheduler, DDIMPipeline, DDPMPipeline
import torch

from model.unet1d import unet
from data.processing import train_dataset2
from model.config import config2

model = unet(grasp_dim=17, cond_dim=10, mid_dim=64, time_dim=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
noise_scheduler = DDIMScheduler(
    num_train_timesteps=config2.num_train_timesteps,
    prediction_type="sample",
    clip_sample=False,
)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config2.learning_rate,
)
train_data2 = train_dataset2(cand_list) 
train_dataloader2 = DataLoader(train_data2, batch_size=config2.train_batch_size, shuffle=True)
model.train()

for e in range(config2.num_epochs):
    loss_sum = 0
    for grasp, block in train_dataloader2:
        grasp = grasp.to(device)
        block = block.to(device)

        B = grasp.shape[0]
        
        noise = torch.randn_like(grasp)
        noise[..., 1:] = 0

        noise_timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (B,),
            device=device,
        ).long()

        noise_grasp = noise_scheduler.add_noise(
            grasp,
            noise,
            noise_timesteps,
        )

        p_cond = 0.1
        save = (torch.rand(B, 1, 1) > p_cond) # broadcast in torch.where to remove all dims of the conditions.
        cfg_cond = torch.where(save, block, torch.zeros_like(block))
        noise_predict = model(noise_grasp, cfg_cond, noise_timesteps)

        loss_mask2 = torch.zeros(B, 1, 4)
        loss_mask2[..., 0] = 1

        optimizer.zero_grad()

        loss = F.mse_loss(
            noise_predict * loss_mask2,
            grasp * loss_mask2,
        ) * grasp.shape[-1]
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    if (e+1) % config2.print_epoch_loss == 0:
        print("loss of an epoch ",e, ": ", loss_sum)

    if e % config2.save_model_epochs == 0:
        filename = f"{config2.output_dir}/epoch_{e}.pt"
        torch.save({'epoch': e, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'loss':loss_sum}, filename)
        print("model saved.")
