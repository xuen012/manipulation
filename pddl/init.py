import torch
from diffusers import DDIMScheduler, DDPMScheduler, UNet1DModel
from model.unet1d import unet

def initialize(
    checkpoint_path,
    test_block_info,
    method="ddim",
    steps=50,
    guidance_scale=3.0,
    device="auto",
):
    # load model architecture
    time_embed_dim = 64

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("format_version") != 2:
        raise ValueError(
            "Retrain using these scripts: this checkpoint needs normalization and model metadata"
        )

    train_steps = checkpoint["config"]["num_train_timesteps"]

    if not 1 <= steps <= train_steps or method not in ("ddpm", "ddim"):
        raise ValueError(
            f"Choose ddpm/ddim and 1 <= steps <= {train_steps}"
        )

    scheduler_class = (
        DDIMScheduler
        if method == "ddim"
        else DDPMScheduler
    )

    noise_scheduler = scheduler_class(
        num_train_timesteps=train_steps,
        prediction_type=checkpoint["prediction_type"],
        clip_sample=False,
    )
    loaded_model = unet(
        grasp_dim=17,
        cond_dim=10,
        mid_dim=64,
        time_dim=64,
    )

    if checkpoint["model_kind"] == "hugging":
        loaded_model = UNet1DModel.from_config(
            checkpoint["model_config"]
        )

    loaded_model.grasp_model_kind = checkpoint["model_kind"]

    if device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    loaded_model = loaded_model.to(device)
    loaded_model.load_state_dict(checkpoint["model"])
    loaded_model.eval()

    # test parameters
    # guidance_scale is a caller option; the original default remains 3.0.
    noise_scheduler.set_timesteps(
        num_inference_steps=steps,
        device=device,
    )

    num_total = 4
    grasp_dim = 17
    cond_dim = 10
    mid_dim = 64
    time_dim = 64
    test_batch_size = 1

    test_block_cnn = test_block_info.unsqueeze(0)
    uncond_block_cnn = torch.zeros_like(test_block_cnn)

    generated_grasp_cnn = torch.randn(
        test_batch_size,
        grasp_dim,
        num_total,
    )

    return (
        loaded_model,
        noise_scheduler,
        guidance_scale,
        test_block_cnn,
        uncond_block_cnn,
        generated_grasp_cnn,
        checkpoint,
    )
