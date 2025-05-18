import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from ComposerPipeline import ComposerStableDiffusionPipeline
from ComposerUnet import ComposerUNet

# ===== 1. 加载Stable Diffusion 1.5的权重 =====
model_id = "runwayml/stable-diffusion-v1-5"

# 1.1 组件加载
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id, subfolder="safety_checker")
feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# 1.2 自定义 UNet 替换原始 UNet
base_unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
unet_config = base_unet.config  # 传递SD1.5的unet配置
custom_unet = ComposerUNet(**unet_config)

# ===== 2. 构建自定义Pipeline =====
pipe = ComposerStableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=custom_unet,
    scheduler=scheduler,
    safety_checker=safety_checker,
    feature_extractor=feature_extractor,
    image_encoder=None
).to("cuda")
# 推理示例
test_data = {
    "image": torch.rand(1, 3, 512, 512).to("cuda"),  # 随机图像
    "pixel_values": torch.rand(1, 3, 224, 224).to("cuda"),
    "prompt": "A fantasy landscape",
    "color": torch.rand(1, 156).to("cuda"),
    "sketch": torch.rand(1, 3, 512, 512).to("cuda"),
    "instance": torch.rand(1, 3, 224, 224).to("cuda"),
    "depth": torch.rand(1, 3, 512, 512).to("cuda"),
    "intensity": torch.rand(1, 1, 64, 64).to("cuda"),
    "guidance_scale": 7.5,
    "num_inference_steps": 50,
}
output = pipe(image=test_data["image"],
              pixel_values=test_data["pixel_values"],
              prompt=test_data["prompt"],
              color=test_data["color"],
              sketch=test_data["sketch"],
              instance=test_data["instance"],
              depth=test_data["depth"],
              intensity=test_data["intensity"],
              guidance_scale=test_data["guidance_scale"],
              num_inference_steps=test_data["num_inference_steps"],
              )
image = output["images"][0]
image.save("result.png")
# ===== 3. 保存Pipeline到本地 =====
save_dir = "./ComposerStableDiffusion"
pipe.save_pretrained(save_dir)
print(f"Pipeline 保存成功: {save_dir}")
# 额外保存 custom UNet 权重
torch.save(custom_unet.state_dict(), f"{save_dir}/composer_unet.pth")
print("Custom UNet 权重已保存")

# ===== 4. 加载保存的Pipeline =====

# 加载其他组件（用默认 from_pretrained）
pipe.save_custom_pretrained("./ComposerStableDiffusion")
loaded_pipe = ComposerStableDiffusionPipeline.load_custom_pretrained("./ComposerStableDiffusion")

print("Pipeline 加载成功！")
