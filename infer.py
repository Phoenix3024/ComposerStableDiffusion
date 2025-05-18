import torch

from ComposerPipeline import ComposerStableDiffusionPipeline

if __name__ == "__main__":
    # loaded_pipe = ComposerStableDiffusionPipeline.load_custom_pretrained("./ComposerStableDiffusion").to("cuda")
    loaded_pipe = ComposerStableDiffusionPipeline.load_custom_pretrained(load_directory=None).to("cuda")
    print("Custom pipeline created successfully!")
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
    output = loaded_pipe(image=test_data["image"],
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
    image.save("result1.png")
    print("Image saved as result.png!")
    print("Custom model loaded successfully!")
