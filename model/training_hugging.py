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

from model.unet1d import model as hugging_model, hugging_input
from data.processing import train_dataset
from data.storage import load_data
from model.config import (
    training_options,
    prepare_data,
    checkpoint_info,
    validation_loss,
    save_progress,
)
import time


def train(cand_list, config, device="auto"):
    torch.manual_seed(config.seed)

    model = UNet1DModel.from_config(hugging_model.config)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        prediction_type="epsilon",
        clip_sample=False,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    train_data, validation_data, normalizer, split = prepare_data(
        cand_list,
        config,
        sequence_length=32,
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=config.train_batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=config.train_batch_size,
        shuffle=False,
    )


    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * config.num_epochs,
    )

    accelerator = Accelerator(
        cpu=(device == "cpu"),
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.num_processes != 1:
        raise ValueError(
            "Use one Accelerate process for this project trainer"
        )

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    metadata = checkpoint_info(
        config,
        normalizer,
        split,
        "hugging",
        32,
        accelerator.unwrap_model(model),
    )

    print(
        "HF model parameters:",
        metadata["parameter_count"],
        "(optional original architecture)",
    )

    history = []
    best = float("inf")
    started = time.perf_counter()

    model.train()
    global_step = 0

    for n in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )

        progress_bar.set_description(f"Epoch {n}")

        loss_sum = 0.0
        seen = 0

        for grasp, block in train_dataloader:
            batch_size = grasp.shape[0]

            # conditional guidance
            p_drop = 0.1
            mask = (
                torch.rand(
                    batch_size,
                    1,
                    1,
                    device=grasp.device,
                )
                < p_drop
            )

            ctx_block = torch.where(
                mask,
                torch.zeros_like(block),
                block,
            )

            # noise learning
            noise = torch.randn_like(grasp)
            noise[..., 1:] = 0

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=grasp.device,
            ).long()

            noise_grasp = noise_scheduler.add_noise(
                grasp,
                noise,
                timesteps,
            )

            with accelerator.accumulate(model):
                predicted_noise = model(
                    hugging_input(
                        noise_grasp,
                        ctx_block,
                        timesteps,
                    ),
                    timestep=timesteps,
                ).sample

                loss_mask = torch.zeros_like(grasp)
                loss_mask[..., 0] = 1.0

                loss = F.mse_loss(
                    predicted_noise * loss_mask,
                    noise * loss_mask,
                ) * grasp.shape[-1]

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite training loss"
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }

            progress_bar.set_postfix(**logs)
            accelerator.log(
                logs,
                step=global_step,
            )

            global_step += 1

            loss_sum += loss.item() * batch_size
            seen += batch_size

            if (
                config.max_minutes
                and time.perf_counter() - started
                >= config.max_minutes * 60
            ):
                break

        progress_bar.close()

        unwrapped = accelerator.unwrap_model(model)

        val_loss = validation_loss(
            unwrapped,
            validation_loader,
            noise_scheduler,
            accelerator.device,
            "hugging",
            config.seed,
        )

        if accelerator.is_main_process:
            pipeline = DDIMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
            )

            if (
                (n + 1) % config.save_model_epochs == 0
                or n == config.num_epochs - 1
            ):
                pipeline.save_pretrained(
                    config.output_dir
                )