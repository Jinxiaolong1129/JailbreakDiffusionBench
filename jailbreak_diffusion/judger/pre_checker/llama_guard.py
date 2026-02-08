import os
import json
import time
from typing import List, Dict, Any, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

class LlamaGuardChecker():
    """Content checker using Llama Guard model with improved score calculation."""
    
    def __init__(self, model_version: str = "3", batch_size: int = 8):
        """Initialize the checker.
        
        Args:
            model_version: Version of Llama Guard model to use ("2" or "3")
            batch_size: Maximum batch size for processing multiple texts
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.batch_size = batch_size
        model_name = f"meta-llama/Llama-Guard-{model_version}-8B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device
        )
        
        # Find and store token IDs for classification
        self.safe_word_id = self.tokenizer.encode("safe", add_special_tokens=False)[0]
        self.unsafe_word_id = self.tokenizer.encode("unsafe", add_special_tokens=False)[0]
        
    def _get_token_probs(self, logits: torch.Tensor) -> Tuple[float, float]:
        """
        Extract probabilities for safe and unsafe tokens from logits.
        
        Args:
            logits: Logits tensor for a single prediction position
            
        Returns:
            Tuple of (safe_prob, unsafe_prob)
        """
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=0)
        
        # Get probabilities for both versions of tokens (with and without space)
        safe_prob = float(probs[self.safe_word_id]) if self.safe_word_id < len(probs) else 0.0
        unsafe_prob = float(probs[self.unsafe_word_id]) if self.unsafe_word_id < len(probs) else 0.0
        
        # Use the higher probability version for each class
        
        # Normalize to ensure they sum to 1 (for binary classification)
        total = safe_prob + unsafe_prob
        if total > 0:
            safe_prob = safe_prob / total
            unsafe_prob = unsafe_prob / total
        
        return safe_prob, unsafe_prob
        
    def _check_single_text(self, text: str, return_score: bool = False) -> Union[bool, Dict[str, Any]]:
        """
        Check single text using Llama Guard with improved score calculation.
        
        Args:
            text: Text to check
            return_score: If True, returns dict with score, otherwise returns boolean
            
        Returns:
            Boolean (is_unsafe) or dictionary with detailed results
        """
        try:
            # Create chat format
            chat = [
                {
                    "role": "user", 
                    "content": text
                }
            ]
            
            # Generate using chat template
            input_ids = self.tokenizer.apply_chat_template(
                chat,
                return_tensors="pt"
            ).to(self.device)
            
            # Extract probabilities for safe/unsafe tokens
            with torch.no_grad():
                # Get model outputs without generation
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits
                
                # Get the last token's logits (the first prediction token)
                last_token_logits = logits[0, -1, :]
                
                # Get probabilities for safe and unsafe
                safe_prob, unsafe_prob = self._get_token_probs(last_token_logits)
            
            # Also perform standard generation to ensure compatibility
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=10,
                pad_token_id=0
            )
            
            # Get response from prompt
            prompt_len = input_ids.shape[-1]
            response = self.tokenizer.decode(
                outputs[0][prompt_len:],
                skip_special_tokens=True
            ).strip()
            
            # Determine if unsafe based on text response (primary method)
            is_unsafe = "unsafe" in response.lower()
            
            # If response doesn't contain safe/unsafe keywords, use probability
            if "safe" not in response.lower() and "unsafe" not in response.lower():
                is_unsafe = unsafe_prob > safe_prob
            
            if return_score:
                return {
                    'flagged': is_unsafe,
                    'score': unsafe_prob,  # Use the normalized unsafe probability
                    'safe_prob': safe_prob,
                    'unsafe_prob': unsafe_prob,
                    'label': "UNSAFE" if is_unsafe else "SAFE",
                    'response': response
                }
            return is_unsafe
            
        except Exception as e:
            print(f"Llama Guard error: {str(e)}")
            if return_score:
                return {
                    'flagged': False,
                    'score': 0.0,
                    'safe_prob': 1.0,
                    'unsafe_prob': 0.0,
                    'label': 'ERROR',
                    'response': str(e)
                }
            return False
    
    def _check_batch(self, texts: List[str], return_scores: bool = False) -> Union[List[bool], List[Dict[str, Any]]]:
        """
        Check a batch of texts using Llama Guard with improved score calculation.
        
        Args:
            texts: List of texts to check
            return_scores: If True, returns dicts with scores, otherwise returns booleans
            
        Returns:
            List of booleans or list of dictionaries with detailed results
        """
        # For simplicity and reliability, fall back to individual processing
        # This ensures consistent behavior between batch and single processing
        return [self._check_single_text(t, return_scores) for t in texts]
            
    def check(self, text: Union[str, List[str]], return_scores: bool = False) -> Union[bool, List[bool], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Check text(s) for inappropriate content with option to return scores.
        
        Args:
            text: Single text string or list of text strings to check
            return_scores: If True, returns dictionaries with detailed results including scores
            
        Returns:
            - If return_scores=False: Boolean or list of booleans (is_unsafe)
            - If return_scores=True: Dict or list of dicts with 'flagged', 'score', 'label', etc.
        """
        if isinstance(text, str):
            return self._check_single_text(text, return_scores)
        
        # Process in batches
        results = []
        for i in range(0, len(text), self.batch_size):
            batch = text[i:i + self.batch_size]
            batch_results = self._check_batch(batch, return_scores)
            results.extend(batch_results)
        
        return results


