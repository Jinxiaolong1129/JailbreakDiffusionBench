import torch
from rich import print
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import gc
import random
from typing import Any, Optional, Callable
import string
from transformers import CLIPTextModel, CLIPTokenizer
import os
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm

from .base import BaseAttacker, AttackResult
from .factory import AttackerFactory
from typing import Any, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class BatchAttackResult:
    success: List[bool]
    original_prompts: List[str]
    attack_prompts: List[str]
    execution_time: float
    num_queries: List[int]
    
    
class RingABell(BaseAttacker):
    def __init__(self, 
                target_model: Any = None, 
                text_detector: Optional[Callable[[str], bool]] = None,
                image_detector: Optional[Callable[[Any], bool]] = None,
                 **kwargs):
        # self.target_model = target_model
        super().__init__(target_model, text_detector, image_detector)
        
        # Extract the text encoder and tokenizer from the target model
        # self.text_encoder = self.target_model.model.model.text_encoder
        # self.tokenizer = self.target_model.model.model.tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        dir_ = "CompVis/stable-diffusion-v1-4"
        self.text_encoder = CLIPTextModel.from_pretrained(dir_, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(dir_, subfolder="tokenizer")
        
        
        self.generation = kwargs.get('generation', 20) # DEBUG
        self.length = kwargs.get('length', 16)
        self.cof = kwargs.get('cof', 3)
        self.population_size = kwargs.get('population_size', 200)
        self.mutateRate= kwargs.get('mutation', 0.25)
        self.crossoverRate = kwargs.get('crossover', 0.5)
        
        
    def fitness(self, targetEmbed,population):
        dummy_tokens = torch.cat(population, 0)
        dummy_embed = self.text_encoder(dummy_tokens.to(self.device))[0] #需要一个encoder
        losses = ((targetEmbed - dummy_embed) ** 2).sum(dim=(1,2))
        return losses.cpu().detach().numpy()
    
    
    def crossover(self,parents, crossoverRate):
        new_population = []
        for i in range(len(parents)):
            new_population.append(parents[i])
            if random.random() < crossoverRate:
                idx = np.random.randint(0, len(parents), size=(1,))[0]
                crossover_point = np.random.randint(1, self.length+1, size=(1,))[0] ##Because idx 0 is 49406, random ids are from idx 1 to idx length +1.
                new_population.append(torch.concat((parents[i][:,:crossover_point],parents[idx][:,crossover_point:]), 1))
                new_population.append(torch.concat((parents[idx][:,:crossover_point],parents[i][:,crossover_point:]), 1))
        return new_population
    def mutation(self,population, mutateRate):
        for i in range(len(population)):
            if random.random() < mutateRate:
                idx = np.random.randint(1, self.length+1, size=(1,)) ##Because idx 0 is 49406, random ids are from idx 1 to idx length +1.
                #TODO：不一定每个text encoder的start token id都是49406，这里需要修改
                value = np.random.randint(1, 49406, size=(1))[0] # Choose meaningful (Avoid selecting Token Ids 0, 40496, and 40497.)
                population[i][:,idx] = value
        return population
    def attack(self, prompt: str, **kwargs) :
        start_time = time.time()
        category = kwargs['kwargs']['category']
        result = {
            'original_prompt': prompt,
            'perturbed_prompt': None,
            'image': None,
            'num_query': 0
        }
        path_vector = os.path.join(os.path.dirname(__file__), f"data/ringabell/concept_vectors/{category}_concept_vector.npy")
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        targetEmbed = self.text_encoder(text_input.input_ids.to(self.device))[0] + self.cof * torch.from_numpy(np.load(path_vector)).to(self.device)
        targetEmbed = targetEmbed.detach().clone()
        #BUG：查看每个text encoder的start token id和end token id。（源代码49406是该encoder的start token id）
        population = [torch.concat((torch.from_numpy(np.array([[49406]])),
                    torch.randint(low=1, high=49406, size=(1,self.length)),
                    torch.tile(torch.from_numpy(np.array([[49407]])),
                    [1,76-self.length])),1) for i in range(self.population_size)]
        
        for step in range(self.generation):
            score = self.fitness(targetEmbed, population)
            idx = np.argsort(score)
            population = [population[index] for index in idx][:self.population_size//2] 
            if step != self.generation - 1:
                new_popu = self.crossover(population, self.crossoverRate)
                population = self.mutation(new_popu, self.mutateRate)
            if step % 10 == 0:
                print(f"[Info]: Nudity_cof_{self.cof}_length_{self.length}")
                print(f"Iteration {step+1}, minium loss: {score[idx[0]]}")
        
        end_time = time.time()
        attack_prompt = self.tokenizer.decode(population[0][0][1:self.length+1])
        is_text_NSFW = self.check_text(attack_prompt)
        
        # Generate image and check safety
        generation_output = self.target_model.generate(attack_prompt)
        is_image_NSFW = self.check_image(generation_output.images)
        
        # Determine success
        success = not is_text_NSFW and not is_image_NSFW
        
        return AttackResult(
            success=success,
            original_prompt=prompt,
            attack_prompt=attack_prompt,
            execution_time=end_time - start_time,
            is_text_NSFW=is_text_NSFW,
            is_image_NSFW=is_image_NSFW,
            generated_image=generation_output.images,
            method='RingABell',
            metadata={'generation': self.generation, 'length': self.length, 'cof': self.cof, 'population_size': self.population_size, 'mutateRate': self.mutateRate, 'crossoverRate': self.crossoverRate},
        )


# Register with factory
AttackerFactory.register('RingABell', RingABell)