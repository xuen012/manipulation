from diffusers import UNet1DModel, DDPMScheduler, DDIMScheduler, DDIMPipeline, DDPMPipeline
import torch

batch_size = 1
num_total = 32
grasp_dim = 17      # output
block_dim = 10       # condition

# in_channels = output + condition dims
model = UNet1DModel(
    in_channels=grasp_dim + block_dim,
    out_channels=grasp_dim,
    block_out_channels=(64, 256), 
    
    # 4. UNet + Self-Attention
    down_block_types=(
        "DownBlock1D",
        "AttnDownBlock1D"
    ),
    up_block_types=(
        "AttnUpBlock1D", # corresponding decoder layer
        "UpBlock1D"
    ),
)

class unet(nn.Module):
    def __init__(self, grasp_dim, cond_dim, mid_dim, time_dim):
        super().__init__()

        # time encoding
        self.time_dim = 64
        self.time_proj = Timesteps(
                time_embed_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0
            )

        #unet
        self.cond = nn.Conv1d(cond_dim, time_dim, kernel_size=1)
        in_dim = grasp_dim + time_dim + time_dim
        self.down = nn.Conv1d(in_dim, mid_dim, 1)
        self.mid_attn = nn.MultiheadAttention(mid_dim, num_heads=2, batch_first=True)
        self.up = nn.Conv1d(mid_dim, 64, 1)
        self.last = nn.Conv1d(64, grasp_dim, 1)

    def forward(self, noise_grasp, cond, t):
        cond_emb = self.cond(cond)
        _, _, length = noise_grasp.shape
        t_emb = self.time_proj(t).unsqueeze(-1).repeat(1, 1, length)
        model_input = torch.cat([torch.tensor(noise_grasp), torch.tensor(cond_emb), torch.tensor(t_emb)], dim=1)
        x = self.down(model_input)
        x = F.silu(x)
        attn_in = x.permute(0, 2, 1)
        attn_out, _ = self.mid_attn(attn_in, attn_in, attn_in)
        x = attn_out.permute(0, 2, 1)
        x = self.up(x)
        x = F.silu(x)
        x = self.last(x)

        return x
