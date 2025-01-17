import torch
from rich import print
from diffusers import StableDiffusionInpaintPipeline
from diffusers import DiffusionPipeline
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import gc
import random
from typing import Any, Optional, Callable
import string
import os
from .base import BaseAttacker, AttackResult
from .factory import AttackerFactory




class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        cos_sim = nn.functional.cosine_similarity(x, y, dim=1, eps=1e-6)
        loss = 1 - cos_sim

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class MMA_Agent:
    """ Manages adversarial prompt attacks. """

    def __init__(self, model, tokenizer, target_embeddings=None, tokens_to_remove_set=None):
        self.model = model
        self.tokenizer = tokenizer
        self.target_embeddings = target_embeddings
        self.tokens_to_remove_set = tokens_to_remove_set
        self.init_control_str()

    def init_control_str(self):
        letters = [random.choice(string.ascii_letters) for _ in range(20)]
        self.control_str = self.best_control = " ".join(letters)

    def token_gradient(self, control, target_embeddings):
        #BUG:查看max_length的值为77
        tokens = self.tokenizer(control, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        input_ids = tokens["input_ids"].cuda()
        embed_weights = self.model.text_model.embeddings.token_embedding.weight

        control_length = 20
        one_hot = torch.zeros(control_length, embed_weights.shape[0], device=self.model.device, dtype=embed_weights.dtype)
        one_hot.scatter_(1, input_ids[0][:control_length].unsqueeze(1), 1.0)
        one_hot.requires_grad_()

        input_embeds = (one_hot @ embed_weights).unsqueeze(0)
        embeds = self.model.text_model.embeddings.token_embedding(input_ids)
        full_embeds = torch.cat([input_embeds, embeds[:, control_length:]], dim=1)

        position_embeddings = self.model.text_model.embeddings.position_embedding
        position_ids = torch.arange(0, 77).cuda()
        pos_embeds = position_embeddings(position_ids).unsqueeze(0)

        embeddings = full_embeds + pos_embeds
        #BUG:
        #阅读这个link：https://github.com/cure-lab/MMA-Diffusion?tab=readme-ov-file,
        embeddings = self.model(input_ids=input_ids, input_embed=embeddings)["pooler_output"]

        criteria = CosineSimilarityLoss()
        loss = criteria(embeddings, self.target_embeddings)
        loss.backward()

        return one_hot.grad.clone()

    @torch.no_grad()
    def logits(self, model, tokenizer, test_controls=None, return_ids=False):
        #BUG:查看max_length的值为77
        cand_tokens = tokenizer(test_controls, padding='max_length', max_length=77, return_tensors="pt", truncation=True)
        input_ids = cand_tokens['input_ids'].cuda()

        if return_ids:
            return model(input_ids=input_ids)['pooler_output'].cuda(), input_ids
        else:
            return model(input_ids=input_ids)['pooler_output'].cuda()

    def sample_control(self, grad, batch_size, topk=256, tokenizer=None, control_str=None):
        for input_id in set(self.tokens_to_remove_set):
            grad[:, input_id] = np.inf

        top_indices = (-grad).topk(topk, dim=1).indices
        tokens = tokenizer.tokenize(control_str)
        control_toks = torch.Tensor(tokenizer.convert_tokens_to_ids(tokens)).to(grad.device).type(torch.int64)

        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(0, len(control_toks), len(control_toks)/batch_size).type(torch.int64).cuda()
        new_token_val = torch.gather(top_indices[new_token_pos], 1, torch.randint(0, topk, (batch_size, 1), device=grad.device))

        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks

    def get_filtered_cands(self, control_cand, filter_cand=True, curr_control=None):
        cands = []
        for i in range(control_cand.shape[0]):
            decoded = self.tokenizer.convert_ids_to_tokens(control_cand[i])
            decoded_str = "".join(decoded).replace('</w>', ' ')[:-1]
            if filter_cand:
                if decoded_str != curr_control and len(self.tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i]):
                    cands.append(decoded_str)
            else:
                cands.append(decoded_str)

        if filter_cand:
            cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
        return cands

    def step(self, batch_size=512, topk=256, filter_cand=True):
        control_cands = []
        new_grad = self.token_gradient(self.control_str, self.target_embeddings)
        new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            control_cand = self.sample_control(new_grad, batch_size, topk, tokenizer=self.tokenizer, control_str=self.control_str)
            control_cands.append(self.get_filtered_cands(control_cand, filter_cand=filter_cand, curr_control=self.control_str))

        del new_grad, control_cand
        gc.collect()

        with torch.no_grad():
            for cand in control_cands:
                cand_embeddings = self.logits(self.model, self.tokenizer, test_controls=cand)
                cos_loss = CosineSimilarityLoss(reduction=None)
                loss = cos_loss(cand_embeddings, self.target_embeddings)

        min_idx = loss.argmin()
        next_control = cand[min_idx]
        cand_loss = loss[min_idx]

        del control_cands, loss, cand
        gc.collect()

        return next_control, cand_loss.item()

    def run(self, n_steps=1000, batch_size=512, topk=256, filter_cand=True):
        best_loss = float('inf')
        for _ in range(n_steps):
            torch.cuda.empty_cache()
            control, loss = self.step(batch_size=batch_size, topk=topk, filter_cand=filter_cand)

            if loss < best_loss:
                best_loss = loss
                self.best_control = control
                print("Best Control:", self.best_control, "Best Loss:", best_loss)

        return self.best_control, best_loss


class MMA(BaseAttacker):
    def __init__(self, target_model: Any = None, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_model = StableDiffusionInpaintPipeline.from_pretrained(
            "benjamin-paine/stable-diffusion-v1-5-inpainting",
            variant="fp16",
            torch_dtype=torch.float16,
        ).to(device)

        self.text_encoder = self.target_model.text_encoder
        self.tokenizer = self.target_model.tokenizer
        self.remove_set()
        self.n_steps = kwargs.get('n_steps', 1000)
        self.topk = kwargs.get('topk', 256)
        self.batch_size = kwargs.get('batch_size', 512)

    def remove_set(self):
        nsfw_word_list = [
            'sexual', 'nude', 'nudity', 'sex', '18+', 'naked', 'nsfw', 'porn', 'dick',
            'vagina', 'naked child', 'explicit content', 'uncensored', 'fuck',
            'nipples', 'visible nipples', 'breast', 'areola',
        ]
        sensitive_word_list = ['ak', 'na', 'ke', 'nud'] + nsfw_word_list

        self.tokens_to_remove_set = set(
            token_id
            for word in sensitive_word_list
            for token, token_id in self.tokenizer.get_vocab().items()
            if word in token
        )
        
    def attack(self, prompt: str, **kwargs) -> AttackResult:
        start_time = time.time()
        result = {
            'original_prompt': prompt,
            'perturbed_prompt': None,
            'image': None,
            'num_query': 0
        }
        #BUG:查看max_length的值为77
        target_tokenized = self.tokenizer(prompt, 
                padding="max_length", max_length=77, 
                return_tensors="pt", truncation=True)
        target_input = target_tokenized["input_ids"].to(self.text_encoder.device)
        target_embedding = self.text_encoder(target_input)["pooler_output"]
        torch.save(target_embedding,"target_embedding.pt")
        '''
            这里保存再读取的原因是因为target_embedding如果不这样处理的话，就会被多次backwards而报错。
            而保存再读取可以避免这个问题。
        '''
        target_embedding = torch.load("target_embedding.pt").to(self.text_encoder.device)
        attack_agent = MMA_Agent(
            model=self.text_encoder,
            tokenizer=self.tokenizer,
            target_embeddings=target_embedding,
            tokens_to_remove_set=self.tokens_to_remove_set
        )

        best_perturbed_prompt, best_loss = attack_agent.run(
            n_steps=self.n_steps, topk=self.topk, batch_size=self.batch_size
        )
        result['perturbed_prompt'] = best_perturbed_prompt
        end_time = time.time()
        is_text_NSFW = self.check_text(result['perturbed_prompt'])
        
        generation_output = self.target_model.generate(result['perturbed_prompt'])
        is_image_NSFW = self.check_image(generation_output.images)
        
        success = bypass_detector and bypass_detector
        return AttackResult(
            success=True,
            original_prompt=prompt,
            attack_prompt=result['perturbed_prompt'],
            execution_time=end_time - start_time,
            bypass_detector=bypass_detector,
            bypass_detector=bypass_detector,
            generated_image=generation_output.images,
            method='MMA',
            metadata={'num_queries': 0},  # Placeholder
        )

AttackerFactory.register('MMA', MMA)