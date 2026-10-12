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
from data.storage import load_data
from model.config import config2, training_options, prepare_data, checkpoint_info, validation_loss, save_progress
import time


def train(cand_list, config2=config2, device="auto"):
    torch.manual_seed(config2.seed)

    model = unet(
        grasp_dim=17,
        cond_dim=10,
        mid_dim=64,
        time_dim=64,
    )

    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

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

    train_data2, validation_data, normalizer, split = prepare_data(
        cand_list,
        config2,
    )

    train_dataloader2 = DataLoader(
        train_data2,
        batch_size=config2.train_batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=config2.train_batch_size,
    )

    metadata = checkpoint_info(
        config2,
        normalizer,
        split,
        "custom",
        4,
        model,
    )

    if metadata["parameter_count"] >= 1_000_000:
        raise ValueError(
            "The primary Project 2 model must be smaller than one million parameters"
        )

    print("Parameters:", metadata["parameter_count"])

    os.makedirs(config2.output_dir, exist_ok=True)

    history = []
    best = float("inf")
    started = time.perf_counter()

    model.train()

    for e in range(config2.num_epochs):
        loss_sum = 0
        seen = 0

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

            save = (
                torch.rand(
                    B,
                    1,
                    1,
                    device=device,
                )
                > p_cond
            )

            cfg_cond = torch.where(
                save,
                block,
                torch.zeros_like(block),
            )

            noise_predict = model(
                noise_grasp,
                cfg_cond,
                noise_timesteps,
            )

            loss_mask2 = torch.zeros(
                B,
                1,
                4,
                device=device,
            )

            loss_mask2[..., 0] = 1

            optimizer.zero_grad()

            loss = F.mse_loss(
                noise_predict * loss_mask2,
                grasp * loss_mask2,
            ) * grasp.shape[-1]

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")

            loss.backward()

            clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            loss_sum += loss.item() * B
            seen += B

            if (
                config2.max_minutes
                and time.perf_counter() - started
                >= config2.max_minutes * 60
            ):
                break

        loss_sum /= seen

        val_loss = validation_loss(
            model,
            validation_loader,
            noise_scheduler,
            device,
            "custom",
            config2.seed,
        )

        elapsed = time.perf_counter() - started

        history.append(
            {
                "epoch": e + 1,
                "train_loss": loss_sum,
                "validation_loss": val_loss,
                "elapsed_seconds": elapsed,
            }
        )

        state = {
            "epoch": e,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss_sum,
            "validation_loss": val_loss,
            **metadata,
        }

        best = save_progress(
            config2.output_dir,
            state,
            history,
            best,
        )

        if (e + 1) % config2.print_epoch_loss == 0:
            print(
                "loss of an epoch ",
                e,
                ": ",
                loss_sum,
            )

        if e % config2.save_model_epochs == 0:
            filename = f"{config2.output_dir}/epoch_{e}.pt"
            torch.save(state, filename)
            print("model saved.")

        if (
            config2.max_minutes
            and elapsed >= config2.max_minutes * 60
        ):
            print(
                "Training time budget reached; best.pt and last.pt saved."
            )
            break

    return model


if __name__ == "__main__":
    args, config = training_options()
    train(
        load_data(args.data),
        config,
        args.device,
    )