import torch
from torch.utils.data import DataLoader

from ComposerUnet import ComposerDataset
from infer import ComposerStableDiffusionPipeline

if __name__ == '__main__':
    composer_pipe = ComposerStableDiffusionPipeline.load_custom_pretrained(load_directory=None).to("cuda")
    # 初始化流程
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建虚拟数据加载器
    dataset = ComposerDataset(num_samples=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 简化训练循环
    optimizer = torch.optim.AdamW(composer_pipe.unet.parameters(), lr=1e-5)
    composer_pipe.set_progress_bar_config(disable=True)  # 禁用进度条显示

    for epoch in range(2):  # 简单运行2个epoch
        # 训练循环调整（示例片段）
        for batch in dataloader:
            # 前向调用需要传递所有条件
            images = batch["image"].to(device)

            latents = composer_pipe.vae.encode(images).latent_dist.sample() * 0.18215

            # 添加噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],)).to(device)
            noisy_latents = composer_pipe.scheduler.add_noise(latents, noise, timesteps)
            noise_pred = composer_pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=None,  # 使用CLIP生成的新context
                image=batch["pixel_values"],
                prompt=batch["prompt"],
                color=batch["color"],
                sketch=batch["sketch"],
                instance=batch["instance"],
                depth=batch["depth"],
                intensity=batch["intensity"]
            )[0]
            # Apply classifier-free guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)
            print(f"noise_pred shape: {noise_pred.shape},noise shape: {noise.shape}")
            # 计算损失
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Loss: {loss.item():.4f}")

    print("Training complete!")
    # 保存模型
    save_directory = "./ComposerStableDiffusion"
    composer_pipe.save_pretrained(save_directory)
    print(f"Model saved to{save_directory}")
