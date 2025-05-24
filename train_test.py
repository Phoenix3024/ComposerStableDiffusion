import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from torch import nn
from torch.utils.data import Dataset, DataLoader


# 修正后的UNet实现
class ModifiedUNet(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 修改输出通道为4以匹配潜变量维度
        self.depth_conv = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # 关键修改处

    def forward(self, sample, timestep, encoder_hidden_states, depth_map=None):
        print(f"Sample shape: {sample.shape}")  # (batch,4,64,64)
        print(f"Sample dtype: {sample.dtype}")  # torch.float16
        if depth_map is not None:
            # 确保深度图与潜变量分辨率一致
            depth_feat = self.depth_conv(depth_map)
            sample = sample + depth_feat  # 现在维度匹配 (batch,4,64,64)


        return super().forward(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states
        )


# 生成随机数据的虚拟数据集
class DummyDataset(Dataset):
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.prompts = ["a cat wearing sunglasses"] * num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            # 生成归一化到[-1,1]的随机图像 (3,512,512)
            "image": torch.rand(3, 512, 512) * 2 - 1,
            # 生成匹配潜变量分辨率的深度图 (1,64,64)
            "depth_map": torch.rand(1, 64, 64) * 2 - 1,
            "prompt": self.prompts[idx]
        }


# 初始化流程
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
original_unet = pipe.unet

# 替换修改后的UNet
pipe.unet = ModifiedUNet(**original_unet.config)
pipe.unet.load_state_dict(original_unet.state_dict(), strict=False)
pipe = pipe.to(device)

# 创建虚拟数据加载器
dataset = DummyDataset(num_samples=20)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 简化训练循环
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-5)
pipe.set_progress_bar_config(disable=True)  # 禁用进度条显示

for epoch in range(2):  # 简单运行2个epoch
    for batch in dataloader:
        # 准备数据
        images = batch["image"].to(device, dtype=torch.float16)
        depth_maps = batch["depth_map"].to(device, dtype=torch.float16)
        prompts = batch["prompt"]

        # VAE编码（使用fp16加速）
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215

            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],)).to(device)
            noisy_latents = pipe.lr_scheduler.add_noise(latents, noise, timesteps)

            # 文本编码
            text_input = pipe.tokenizer(
                prompts,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt"
            )
            text_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]
            print(f"Text embeddings shape: {text_embeddings.shape}")  # (batch_size, seq_len, embed_dim)

            # 前向传播
            noise_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
                depth_map=depth_maps
            ).sample

            # 计算损失
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Loss: {loss.item():.4f}")

print("代码运行成功！")