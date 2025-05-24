import torch
from torch.utils.data import DataLoader

from ComposerUnet import ComposerDataset
from infer import ComposerStableDiffusionPipeline
from diffusers.optimization import get_scheduler

if __name__ == '__main__':
    composer_pipe = ComposerStableDiffusionPipeline.load_custom_pretrained(load_directory=None).to("cuda")
    # 初始化流程
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建虚拟数据加载器
    dataset = ComposerDataset(num_samples=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    num_epochs = 2

    # 简化训练循环
    optimizer = torch.optim.AdamW(composer_pipe.unet.parameters(), lr=1e-5)

    # 学习率线性增长
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=15000
    )
    composer_pipe.set_progress_bar_config(disable=True)  # 禁用进度条显示

    # 设置为训练模式
    composer_pipe.unet.train()
    composer_pipe.vae.eval()
    scaling_factor = composer_pipe.vae.config.scaling_factor

    for epoch in range(num_epochs):
        # 训练循环调整（示例片段）
        for batch in dataloader:
            # 前向调用需要传递所有条件
            images = batch["image"].to(device)

            # 冻结VAE
            with torch.no_grad():
                latents = composer_pipe.vae.encode(images).latent_dist.sample() * scaling_factor

            # 添加噪声
            noise = torch.randn_like(latents)

            # 计算噪声
            timesteps = torch.randint(0, composer_pipe.lr_scheduler.config.num_train_steps, (latents.shape[0],))
            timesteps = timesteps.to(device)
            timesteps.long()

            # 添加噪声到潜在变量
            noisy_latents = composer_pipe.lr_scheduler.add_noise(latents, noise, timesteps)

            # 计算噪声预测
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
            # print(f"noise_pred shape: {noise_pred.shape},noise shape: {noise.shape}")

            # 计算损失
            loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="mean")

        # 反向传播
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(composer_pipe.unet.parameters(), 1.0)
        optimizer.step()
        # 学习率调度
        lr_scheduler.step()
        # 清除梯度
        optimizer.zero_grad()
        print(f"Loss: {loss.item():.4f}")

    print("Training complete!")
    # 保存模型
    save_directory = "./ComposerStableDiffusion"
    composer_pipe.save_custom_pretrained(save_directory)
    print(f"Model saved to{save_directory}")
