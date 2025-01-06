import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")



# exit()

# from sld import SLDPipeline
# device='cuda'

# pipe = SLDPipeline.from_pretrained(
#     "AIML-TUDA/stable-diffusion-safe",
# ).to(device)


# import torch
# gen = torch.Generator(device)
# prompt = 'portrait of Sickly diseased dying Samurai warrior, sun shining, photo realistic illustration by greg rutkowski, thomas kindkade, alphonse mucha, loish, norman rockwell.'
# gen.manual_seed(2602096847)
# out = pipe(prompt=prompt, generator=gen, guidance_scale=10,
#            sld_warmup_steps=7,
#            sld_guidance_scale=2000,
#            sld_threshold=0.025,
#            sld_momentum_scale=0.5,
#            sld_mom_beta=0.7
#            )
# image = out.images[0]
# # save images
# image.save('test.png')




# from diffusers import CogView3PlusPipeline
# import torch

# pipe = CogView3PlusPipeline.from_pretrained("THUDM/CogView3-Plus-3B", torch_dtype=torch.float16).to("cuda")

# # Enable it to reduce GPU memory usage
# pipe.enable_model_cpu_offload()
# pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()

# prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."
# image = pipe(
#     prompt=prompt,
#     guidance_scale=7.0,
#     num_images_per_prompt=1,
#     num_inference_steps=50,
#     width=1024,
#     height=1024,
# ).images[0]

# image.save("cogview3.png")
