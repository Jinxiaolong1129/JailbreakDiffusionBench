from huggingface_hub import login
login(token="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")

from transformers import AutoModel
model = AutoModel.from_pretrained("AdamCodd/distilroberta-nsfw-prompt-stable-diffusion")
print(model)