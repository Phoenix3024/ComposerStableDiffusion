import os

import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from ComposerUnet import ComposerUNet


class ComposerStableDiffusionPipeline(StableDiffusionPipeline):
    """
    Custom Stable Diffusion pipeline that integrates a MegaConditionUNet combining CLIP image/text features
    with additional local conditions (color, sketch, instance, depth, intensity).
    """

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: ComposerUNet,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: None,
            requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
            safety_checker=safety_checker, feature_extractor=feature_extractor, image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker
        )
        self.register_modules(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
                              safety_checker=safety_checker, feature_extractor=feature_extractor)
        self.logger = logging.get_logger(__name__)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @torch.no_grad()
    def __call__(
            self,
            image: torch.Tensor or Image.Image,
            pixel_values: torch.Tensor,
            prompt: str,
            color: torch.Tensor = None,
            sketch: torch.Tensor = None,
            instance: torch.Tensor = None,
            depth: torch.Tensor = None,
            intensity: torch.Tensor = None,
            guidance_scale: float = 1.0,
            num_inference_steps: int = 50,
    ):
        """
        Generate an image conditioned on text prompt and additional inputs:
        - prompt: text string
        - image: PIL Image or tensor to extract CLIP features
        - color, sketch, instance, depth, intensity: tensors for local conditioning
        """

        batch_size = image.shape[0]

        # Prepare latent noise
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, torch.Tensor):
            _, _, height, width = image.shape
        else:
            height = width = 512

        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            device=self.device,
            dtype=image.dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # Diffusion process
        for t in self.scheduler.timesteps:
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=None,  # 使用CLIP生成的新context
                image=pixel_values,
                prompt=prompt,
                color=color,
                sketch=sketch,
                instance=instance,
                depth=depth,
                intensity=intensity,
            )[0]

            # Apply classifier-free guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            # print(f"noise_pred shape: {noise_pred.shape},latents shape: {latents.shape}")

            # Compute previous noisy sample x_{t-1}
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        recon_image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=None)[0]
        do_denormalize = [True] * recon_image.shape[0]
        recon_image = self.image_processor.postprocess(recon_image, output_type="pil", do_denormalize=do_denormalize)

        return {"images": recon_image}

    def save_custom_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        # 保存 diffusers 标准组件
        super().save_pretrained(save_directory)

        # 保存自定义 UNet 的权重（state_dict）
        torch.save(self.unet.state_dict(), os.path.join(save_directory, "composer_unet.pth"))
        print(f"[保存成功] Pipeline 与自定义 UNet 保存到 {save_directory}")

    @classmethod
    def load_custom_pretrained(cls, load_directory: str or None,
                               base_unet_model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        自定义加载函数：加载 Pipeline + 注入 UNet state_dict。
        参数：
            - load_directory: 保存路径（包含 pipe + composer_unet.pth）
            - base_unet_model_id: 用于获取原始 SD1.5 的 unet config
        """

        # 加载其他组件
        if load_directory is None:
            load_directory = base_unet_model_id
        vae = AutoencoderKL.from_pretrained(load_directory, subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(load_directory, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(load_directory, subfolder="tokenizer")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(load_directory, subfolder="safety_checker")
        feature_extractor = CLIPImageProcessor.from_pretrained(load_directory, subfolder="feature_extractor")
        scheduler = DDPMScheduler.from_pretrained(load_directory, subfolder="scheduler")

        # 构造并加载自定义 UNet
        base_unet = UNet2DConditionModel.from_pretrained(base_unet_model_id, subfolder="unet")
        for config in base_unet.config.keys():
            if config == "time_cond_proj_dim":
                base_unet.config[config] = 320
        custom_unet = ComposerUNet(**base_unet.config)
        if load_directory != "runwayml/stable-diffusion-v1-5":
            state_dict_path = os.path.join(load_directory, "composer_unet.pth")
            custom_unet.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

        # 构造 Pipeline
        pipe = cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=custom_unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=None
        )

        print(f"[加载成功] Pipeline 从 {load_directory} 加载完成")
        return pipe
