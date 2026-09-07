# loaded model architecture
time_embed_dim = 64
noise_scheduler = DDIMScheduler(num_train_timesteps=1000, prediction_type="sample")
loaded_model = unet(grasp_dim=17, cond_dim=10, mid_dim=64, time_dim=64)
checkpoint = torch.load("./epoch_19999.pt", weights_only=False)
loaded_model.load_state_dict(checkpoint['model'])
loaded_model.eval()

# test parameters
guidance_scale = 3.0
loaded_model.eval()
noise_scheduler.set_timesteps(num_inference_steps=1000)
num_total = 4
grasp_dim = 17; cond_dim = 10; mid_dim=64; time_dim=64
test_batch_size = 1
# 給定你要測試的新條件 (B, 9, N)
test_block_cnn = test_block_info.unsqueeze(0)#torch.randn(batch_size, block_dim, num_total)
uncond_block_cnn = torch.zeros_like(test_block_cnn)
generated_grasp_cnn = torch.randn(test_batch_size, grasp_dim, num_total)
