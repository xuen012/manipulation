"""The missing sampling adapter; denoising stays in pddl/evaluate_diffusion.py."""
from collections import Counter
import time
import torch
from data.processing import condition_tensor, denormalize_grasps, candidate_from_vector
from pddl.init import initialize


class GraspSampler:
    def __init__(self, checkpoint, method="ddim", steps=50, guidance_scale=3., device="auto", seed=42):
        (self.loaded_model, self.noise_scheduler, self.guidance_scale, _, _, _,
         self.checkpoint) = initialize(checkpoint, method=method, steps=steps,
                                      guidance_scale=guidance_scale, device=device)
        self.device = next(self.loaded_model.parameters()).device
        self.normalizer = self.checkpoint["normalizer"]
        self.sequence_length = self.checkpoint["sequence_length"]
        self.reseed(seed)
        self.last_sampling_ms = 0.
        self.total_sampling_ms = 0.
        self.total_generated = 0

    def reseed(self, seed):
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def sample(self, pos, half, quat, count=1):
        from pddl.evaluate_diffusion import plan_loop
        if count < 1:
            raise ValueError("count must be positive")
        self.synchronize()
        started = time.perf_counter()
        test_block_info = condition_tensor(pos, half, quat, self.normalizer, self.sequence_length)
        test_block_cnn = test_block_info.unsqueeze(0).repeat(count, 1, 1).to(self.device)
        uncond_block_cnn = torch.zeros_like(test_block_cnn)
        generated_grasp_cnn = torch.randn((count, 17, self.sequence_length), generator=self.generator).to(self.device)
        generated_grasp_cnn[..., 1:] = 0
        final_grasp = plan_loop(self.loaded_model, self.guidance_scale, self.noise_scheduler,
                               uncond_block_cnn, test_block_cnn, generated_grasp_cnn,
                               generator=self.generator)
        result = denormalize_grasps(final_grasp, self.normalizer)
        self.synchronize()
        self.last_sampling_ms = 1000 * (time.perf_counter() - started)
        self.total_sampling_ms += self.last_sampling_ms
        self.total_generated += count
        return result


# TAMPanda stays optional for pure PyTorch data/training and checkpoint loading.
def make_planner(checkpoint, env, **options):
    from model.grasp_planner import DiffusionGraspPlanner
    return DiffusionGraspPlanner(checkpoint, env, **options)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import numpy as np
    parser = argparse.ArgumentParser(description="Sample using the original denoising loop")
    parser.add_argument("--checkpoint", default="checkpoints/grasp/best.pt")
    parser.add_argument("--method", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output", default="results/samples.npz")
    args = parser.parse_args()
    sampler = GraspSampler(args.checkpoint, args.method, args.steps)
    samples = sampler.sample([.4, .4, .29], [.02, .02, .02], [1, 0, 0, 0], args.count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, grasps=samples[:, :, 0].numpy())
    print(f"Saved unvalidated grasp samples to {output}")
