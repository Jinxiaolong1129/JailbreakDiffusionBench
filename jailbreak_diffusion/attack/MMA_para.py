# jailbreak_diffusion/attack/MMA_para.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import string
import gc
from typing import Any, Optional, Callable, List, Dict
from dataclasses import dataclass
import os
from tqdm import tqdm
import numpy as np
from transformers.models.clip import modeling_clip
from .base import BaseAttacker, AttackResult
from .factory import AttackerFactory


@dataclass
class BatchAttackResult:
    success: List[bool]
    original_prompts: List[str]
    attack_prompts: List[str]
    execution_time: float
    num_queries: List[int]


class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        cos_sim = F.cosine_similarity(x, y, dim=1, eps=1e-6)
        loss = 1 - cos_sim

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class CosineSimilarityLoss_Batch(nn.Module):
    def __init__(self, reduction=None):
        super(CosineSimilarityLoss_Batch, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        """
        Args:
            x: [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
            y: [batch_size, feature_dim]
        Returns:
            loss: [batch_size, seq_len] or [batch_size]
        """
        # Handle different input shapes
        if len(x.shape) == 3:  # [batch_size, seq_len, feature_dim]
            # Expand y to [batch_size, 1, feature_dim]
            y = y.unsqueeze(1)
            # Calculate similarity along feature dimension
            cos_sim = F.cosine_similarity(x, y, dim=2, eps=1e-6)
        else:  # [batch_size, feature_dim]
            cos_sim = F.cosine_similarity(x, y, dim=1, eps=1e-6)
        
        # Compute loss
        loss = 1 - cos_sim
        
        # Apply reduction if specified
        if self.reduction == 'mean':
            if len(x.shape) == 3:
                loss = loss.mean(dim=1)  # Average across sequence length
            else:
                loss = loss.mean()
        elif self.reduction == 'sum':
            if len(x.shape) == 3:
                loss = loss.sum(dim=1)  # Sum across sequence length
            else:
                loss = loss.sum()
                
        return loss


class ParallelMMA(BaseAttacker):
    """
    Parallel implementation of Masked Multimodal Attack (MMA) for batch processing
    """
    
    def __init__(
        self, 
        target_model: Any = None, 
        text_detector: Optional[Callable[[str], bool]] = None,
        image_detector: Optional[Callable[[Any], bool]] = None,
        **kwargs
    ):
        super().__init__(target_model, text_detector, image_detector)
        
        # Extract the text encoder and tokenizer from the target model
        self.text_encoder = self.target_model.model.model.text_encoder
        self.tokenizer = self.target_model.model.model.tokenizer
        
        # Initialize parameters
        self.n_steps = kwargs.get('n_steps', 1000)
        self.topk = kwargs.get('topk', 256)
        self.batch_size = kwargs.get('batch_size', 512)
        self.internal_batch_size = kwargs.get('internal_batch_size', 256)
        
        # Initialize the token removal set
        self.remove_set()
        
    def remove_set(self):
        """Initialize the set of tokens to be removed/filtered during sampling"""
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
    
    def get_target_embeddings_batch(self, prompts: List[str]):
        """Compute embeddings for a batch of target prompts"""
        target_tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )
        target_input = target_tokenized["input_ids"].to(self.text_encoder.device)
        
        with torch.no_grad():
            target_embeddings = self.text_encoder.text_model(target_input)["pooler_output"]
        return target_embeddings
    
    def token_gradient_batch(self, controls: List[str], target_embeddings: torch.Tensor):
        """Compute gradients for a batch of control strings"""
        # Save target embeddings to a temporary file and reload to detach from computation graph
        tmp_path = os.path.join(os.path.dirname(__file__), "data/ParallelMMA/target_embedding_tmp.pt")
        os.makedirs(os.path.join(os.path.dirname(__file__), "data/ParallelMMA"), exist_ok=True)
        torch.save(target_embeddings, tmp_path)
        
        # Reload to detach from computation graph
        target_embeddings = torch.load(tmp_path).to(self.text_encoder.device)
        
        tokens = self.tokenizer(
            controls, 
            padding="max_length",
            return_tensors="pt", 
            truncation=True
        )
        input_ids = tokens["input_ids"].to(self.text_encoder.device)
        
        batch_size = len(controls)
        control_length = 20
        embed_weights = self.text_encoder.text_model.embeddings.token_embedding.weight
        
        # Create one-hot vectors for the entire batch
        one_hot = torch.zeros(
            batch_size,
            control_length, 
            embed_weights.shape[0], 
            device=self.text_encoder.device, 
            dtype=embed_weights.dtype
        )
        
        # Batch scatter operation
        for i in range(batch_size):
            one_hot[i].scatter_(1, input_ids[i][:control_length].unsqueeze(1), 1.0)
        
        one_hot.requires_grad_()
        
        # Compute embeddings
        input_embeds = torch.matmul(one_hot, embed_weights)
        embeds = self.text_encoder.text_model.embeddings.token_embedding(input_ids)
        full_embeds = torch.cat(
            [input_embeds, embeds[:, control_length:]], dim=1
        )
        
        # Add position embeddings
        position_ids = torch.arange(0, self.tokenizer.model_max_length).to(self.text_encoder.device)
        position_embeddings = self.text_encoder.text_model.embeddings.position_embedding
        pos_embeds = position_embeddings(position_ids).unsqueeze(0).expand(batch_size, -1, -1)
        embeddings = full_embeds + pos_embeds
        
        # Compute loss and gradients
        control_embeddings = self.text_encoder.text_model(
            input_ids=input_ids, 
            input_embed=embeddings  
        )["pooler_output"]
        
        criteria = CosineSimilarityLoss()
        loss = criteria(control_embeddings, target_embeddings)
        loss.backward()  # No need for retain_graph=True now
        
        return one_hot.grad.clone()
    
    
    def sample_control_batch(self, grad, control_strs, topk=256, num_candidates=512):
        """
        Sample control tokens based on gradient information for a batch.
        
        Args:
            grad: Gradient tensor of shape [batch_size, control_length, vocab_size]
            control_strs: List of current control strings
            topk: Number of top candidates to consider for replacement
            num_candidates: Number of samples to generate per batch item
            
        Returns:
            Tensor of sampled control tokens [batch_size, num_candidates, control_length]
        """
        batch_size, num_tokens, vocab_size = grad.shape
        
        # Mask sensitive tokens
        for input_id in self.tokens_to_remove_set:
            grad[:, :, input_id] = float("inf")
        
        # Get topk indices
        top_indices = (-grad).topk(topk, dim=2).indices  # [batch_size, control_length, topk]
        
        # Tokenize current control strings
        tokenized = self.tokenizer.batch_encode_plus(
            control_strs,
            add_special_tokens=False,
            return_tensors="pt"
        )
        control_toks = tokenized["input_ids"].to(grad.device).type(torch.int64)
        
        # Expand control tokens to create multiple candidates
        original_control_toks = control_toks.unsqueeze(1).expand(batch_size, num_candidates, -1)
        
        # Randomly select positions to modify for each candidate
        new_token_pos = torch.arange(0, control_toks.size(1), control_toks.size(1) / num_candidates).type(torch.int64).to(grad.device)
        new_token_pos = new_token_pos.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, num_candidates]
        
        # Get batch indices for selecting from top_indices
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, new_token_pos.size(1))
        
        # Select indices based on positions
        selected_top_indices = top_indices[
            batch_indices,
            new_token_pos,
            :
        ]  # [batch_size, num_candidates, topk]
        
        # Randomly select from topk for each position
        random_indices = torch.randint(0, topk, (batch_size, num_candidates, 1), device=top_indices.device)
        new_token_val = torch.gather(selected_top_indices, 2, random_indices)  # [batch_size, num_candidates, 1]
        
        # Create new control tokens by replacing at selected positions
        new_control_toks = original_control_toks.clone().scatter_(
            2,  # Dimension to scatter on (token dimension)
            new_token_pos.unsqueeze(-1),  # [batch_size, num_candidates, 1]
            new_token_val  # [batch_size, num_candidates, 1]
        )
        
        return new_control_toks
    
    def get_filtered_cands(self, control_cand, current_controls=None, filter_cand=True):
        """
        Filter and decode candidate tokens.
        
        Args:
            control_cand: Tensor of token IDs [batch_size, num_candidates, control_length]
            current_controls: List of current control strings
            filter_cand: Whether to filter candidates
            
        Returns:
            List[List[str]]: Decoded candidate strings for each batch item
        """
        batch_size, num_candidates, token_length = control_cand.shape
        
        # Reshape for efficient processing
        flat_tokens = control_cand.reshape(-1, token_length).tolist()
        
        # Convert to tokens and join
        candidates_by_batch = []
        for batch_idx in range(batch_size):
            batch_candidates = []
            start_idx = batch_idx * num_candidates
            end_idx = start_idx + num_candidates
            
            for cand_idx in range(start_idx, end_idx):
                decoded_tokens = self.tokenizer.convert_ids_to_tokens(flat_tokens[cand_idx])
                decoded_str = "".join(decoded_tokens).replace('</w>', ' ').strip()
                
                # Filter if needed
                if filter_cand and current_controls:
                    tokenized_len = len(self.tokenizer(decoded_str, add_special_tokens=False).input_ids)
                    expected_len = len(control_cand[batch_idx, cand_idx % num_candidates])
                    
                    if decoded_str != current_controls[batch_idx] and tokenized_len == expected_len:
                        batch_candidates.append(decoded_str)
                else:
                    batch_candidates.append(decoded_str)
            
            # Ensure we have enough candidates
            if filter_cand and batch_candidates:
                batch_candidates = batch_candidates + [batch_candidates[-1]] * (num_candidates - len(batch_candidates))
            elif filter_cand and not batch_candidates:
                # If all filtered out, keep current control
                batch_candidates = [current_controls[batch_idx]] * num_candidates
                
            candidates_by_batch.append(batch_candidates[:num_candidates])  # Ensure fixed size
            
        return candidates_by_batch
    
    def step(self, batch_controls, batch_target_embeddings, topk=256, filter_cand=True, candidate_size=512):
        """
        Perform one optimization step for a batch of controls.
        
        Args:
            batch_controls: List of current control strings
            batch_target_embeddings: Target embeddings tensor
            topk: Number of top candidates to consider
            filter_cand: Whether to filter candidates
            candidate_size: Number of candidates to generate per batch item
            
        Returns:
            Tuple of (next_controls, losses)
        """
        # Calculate gradients
        grads = self.token_gradient_batch(batch_controls, batch_target_embeddings)
        
        # Sample new controls based on gradients
        new_controls = self.sample_control_batch(
            grads, 
            batch_controls,
            topk=topk,
            num_candidates=candidate_size
        )
        
        # Filter and decode candidates
        control_candidates = self.get_filtered_cands(
            new_controls,
            current_controls=batch_controls,
            filter_cand=filter_cand
        )
        
        # Clean up to save memory
        del grads, new_controls
        gc.collect()
        
        # Evaluate candidates (process in smaller batches if needed)
        batch_size = len(batch_controls)
        all_embeddings = []
        for batch_idx in range(batch_size):
            candidates = control_candidates[batch_idx]
            embeddings = self.get_target_embeddings_batch(candidates)
            all_embeddings.append(embeddings)
        
        # Compute loss for all candidates
        with torch.no_grad():
            loss_batch = []
            for batch_idx, embeddings in enumerate(all_embeddings):
                criteria = CosineSimilarityLoss_Batch(reduction=None)
                target_emb = batch_target_embeddings[batch_idx].unsqueeze(0).repeat(len(control_candidates[batch_idx]), 1)
                loss = criteria(embeddings, target_emb)
                loss_batch.append(loss)
        
        # Select best candidate for each batch item
        next_controls = []
        best_losses = []
        
        for batch_idx in range(batch_size):
            min_idx = loss_batch[batch_idx].argmin().item()
            next_controls.append(control_candidates[batch_idx][min_idx])
            best_losses.append(loss_batch[batch_idx][min_idx].item())
        
        return next_controls, torch.tensor(best_losses)
    
    def optimize_batch(self, prompts, target_embeddings, n_steps=1000, batch_size=512, topk=256):
        """
        Optimize a batch of prompts.
        
        Args:
            prompts: List of original prompts
            target_embeddings: Tensor of target embeddings
            n_steps: Number of optimization steps
            batch_size: Size of internal batches to process
            topk: Number of top tokens to consider
            
        Returns:
            List of optimized control strings
        """
        num_prompts = len(prompts)
        num_batches = (num_prompts + batch_size - 1) // batch_size
        results = []
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_prompts)
            
            # Get batch data
            batch_prompts = prompts[start_idx:end_idx]
            batch_embeddings = target_embeddings[start_idx:end_idx]
            
            # Initialize controls with random strings
            batch_controls = [
                " ".join([random.choice(string.ascii_letters) for _ in range(20)])
                for _ in range(len(batch_prompts))
            ]
            
            best_controls = batch_controls.copy()
            best_losses = [float('inf')] * len(batch_controls)
            
            # Optimization loop
            for step in tqdm(range(n_steps), desc=f"Optimizing batch {batch_idx+1}/{num_batches}"):
                # Perform optimization step
                control, loss = self.step(
                    batch_controls=batch_controls,
                    batch_target_embeddings=batch_embeddings,
                    topk=topk
                )
                
                # Update controls
                batch_controls = control
                
                # Track best results
                for i, current_loss in enumerate(loss):
                    if current_loss < best_losses[i]:
                        best_losses[i] = current_loss.item()
                        best_controls[i] = control[i]
                
                # Print progress
                if step % 10 == 0:
                    avg_loss = sum(best_losses) / len(best_losses)
                    print(f"Step {step}/{n_steps} | Avg Best Loss: {avg_loss:.4f}")
                
                # Early stopping check
                if max(best_losses) < 0.1:
                    print(f"Early stopping at step {step} - target loss achieved")
                    break
            
            results.extend(best_controls)
        
        return results
    
    def attack_batch(self, prompts: List[str], **kwargs) -> BatchAttackResult:
        """
        Run attack on multiple prompts in parallel.
        
        Args:
            prompts: List of prompts to attack
            
        Returns:
            BatchAttackResult with attack results
        """
        start_time = time.time()
        
        # Extract parameters
        n_steps = kwargs.get('n_steps', self.n_steps)
        topk = kwargs.get('topk', self.topk)
        batch_size = kwargs.get('batch_size', self.internal_batch_size)
        
        # Get target embeddings for all prompts
        target_embeddings = self.get_target_embeddings_batch(prompts)
        
        # Run optimization
        attack_prompts = self.optimize_batch(
            prompts, 
            target_embeddings, 
            n_steps=n_steps,
            batch_size=batch_size,
            topk=topk
        )
        
        end_time = time.time()
        
        # Initialize success and query counts (actual values would be determined during generation)
        success = [True] * len(prompts)  # Placeholder
        num_queries = [n_steps] * len(prompts)  # Approximate count
        
        return BatchAttackResult(
            success=success,
            original_prompts=prompts,
            attack_prompts=attack_prompts,
            execution_time=end_time - start_time,
            num_queries=num_queries
        )
    
    def attack(self, prompt: str, attack_prompt: str = None, **kwargs) -> AttackResult:
        """
        Run attack on a single prompt. Inherits from BaseAttacker.
        
        Args:
            prompt: Original prompt to attack
            attack_prompt: Optional pre-generated attack prompt
            
        Returns:
            AttackResult with attack results
        """
        start_time = time.time()
        
        if not attack_prompt:
            print("ParallelMMA | Generating attack prompt")
            
            # Get target embedding
            target_tokenized = self.tokenizer(
                prompt,
                padding="max_length",
                return_tensors="pt", 
                truncation=True
            )
            target_input = target_tokenized["input_ids"].to(self.text_encoder.device)
            target_embedding = self.text_encoder.text_model(target_input)["pooler_output"]
            
            # Initialize control string
            control_str = " ".join([random.choice(string.ascii_letters) for _ in range(20)])
            best_control = control_str
            best_loss = float('inf')
            
            # Optimization loop
            for step in range(self.n_steps):
                # Create a batch with just one prompt
                control, loss = self.step(
                    batch_controls=[control_str],
                    batch_target_embeddings=target_embedding,
                    topk=self.topk
                )
                
                control_str = control[0]
                current_loss = loss[0].item()
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_control = control_str
                    print(f"Step {step}/{self.n_steps} | Best Control: {best_control} | Loss: {best_loss:.4f}")
                
                if step % 10 == 0:
                    print(f"Step {step}/{self.n_steps} | Current Control: {control_str} | Loss: {current_loss:.4f}")
                
                # Early stopping check
                if best_loss < 0.1:
                    print(f"Early stopping at step {step} - target loss achieved")
                    break
            
            attack_prompt = best_control
        else:
            print(f"Using provided attack prompt: {attack_prompt}")
        
        end_time = time.time()
        
        # Check text safety
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
            method='ParallelMMA',
            metadata={'num_queries': self.n_steps},  # Approximate count
        )


# Register with factory
AttackerFactory.register('ParallelMMA', ParallelMMA)