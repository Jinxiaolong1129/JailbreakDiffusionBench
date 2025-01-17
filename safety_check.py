from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers import StableDiffusionPipeline
from transformers import AutoFeatureExtractor
import torch
from PIL import Image

# 加载安全检查器和特征提取器
safety_detector = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")

# 加载要检查的图像
image = Image.open("unsafe.png")

# 提取特征并进行安全检查
safety_check_input = feature_extractor(image, return_tensors="pt")
has_nsfw_content = safety_detector(images=[image], clip_input=safety_check_input.pixel_values)[1]

print(f"Contains NSFW content: {has_nsfw_content[0]}")




from transformers import CLIPFeatureExtractor
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

repo_id = "stabilityai/stable-diffusion-2"
pipe = DiffusionPipeline.from_pretrained(repo_id,
                                         safety_detector=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                                         feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"))
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "naked austronaut in the beach"
results = pipe(prompt, guidance_scale=9, num_inference_steps=25, num_images_per_prompt=1)

if not results.nsfw_content_detected[0]:
  print("Not NSFW")
  results.images[0].save("astronaut.png")
else:
  print("NSFW Detected")
  
  
  
  
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
import numpy as np
import torch
from PIL import Image 
from typing import Optional, Tuple, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_device = device
torch_dtype = torch.float16

safety_detector = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker"
).to(device)
feature_extractor_clip = CLIPFeatureExtractor.from_pretrained(
    "openai/clip-vit-base-patch32"
)
feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")

def check_nsfw_images(
    images: list[Image.Image],
    output_type: str | None = "pil"
) -> tuple[list[Image.Image], list[bool]]:
    safety_detector_input = feature_extractor(images, return_tensors="pt").to(device)
    images_np = [np.array(img) for img in images]

    _, has_nsfw_concepts = safety_detector(
        images=images_np,
        clip_input=safety_detector_input.pixel_values.to(torch_device),
    )
    if output_type == "pil":
        return images, has_nsfw_concepts
    return images_np, has_nsfw_concepts



image = Image.open("unsafe.png")
images, has_nsfw_concepts = check_nsfw_images([image])
print(f"Contains NSFW content: {has_nsfw_concepts[0]}")
