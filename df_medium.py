
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=28,
    guidance_scale=4.5,
).images[0]
image.save("capybara.png")


import torch
from diffusers import StableDiffusion3Pipeline, DiffusionPipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16,
                                                device_map="balanced")
# pipe = pipe.to("cuda")

prompts = [
    "A capybara holding a sign that reads Hello World",
    # "A red panda playing a guitar on a mountain top",
    # "An astronaut riding a horse on Mars, photorealistic"
] 

print("Generating image...")
print(len(prompts))
image = pipe(
    prompts,
    negative_prompt="",
    num_inference_steps=28,
    guidance_scale=7.0,
)

print(image)

