from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import logging
import warnings
import time
import json
import hashlib

from .cache import SQLiteCache

logger = logging.getLogger(__name__)


class LocalLlamaClient:
    """Local Llama-3.1-8B-Instruct client using HuggingFace Transformers.
    
    Adapted for CPU-only environments. For GPU environments, consider using
    quantization via bitsandbytes.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        use_cpu: bool = False,
        cache: Optional[SQLiteCache] = None,
    ):
        self.model_name = model_name
        self.cache = cache
        
        # Determine device
        if use_cpu or not torch.cuda.is_available():
            self.device = "cpu"
            if load_in_4bit or load_in_8bit:
                warnings.warn(
                    "Quantization with bitsandbytes requires GPU. "
                    "Running on CPU without quantization (may be slow)."
                )
                load_in_4bit = False
                load_in_8bit = False
        else:
            self.device = device if device != "auto" else "cuda"
        
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure loading parameters
        kwargs = {}
        
        if self.device == "cpu":
            # CPU: use float32 for stability, or float16 if available
            kwargs["torch_dtype"] = torch.float32
            logger.info("Loading model on CPU (this may be slow)...")
        else:
            # GPU: use bfloat16
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "auto"
            
            # GPU quantization
            if load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info("Using 4-bit quantization")
                except ImportError:
                    logger.warning("bitsandbytes not available, loading without quantization")
            elif load_in_8bit:
                try:
                    kwargs["load_in_8bit"] = True
                    logger.info("Using 8-bit quantization")
                except Exception:
                    logger.warning("8-bit quantization failed, loading without quantization")
        
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs
        )
        
        # Move to device if CPU
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")
        
        # Performance warning for CPU
        if self.device == "cpu":
            logger.warning(
                "⚠️  Running on CPU will be VERY SLOW (~10-30 seconds per generation). "
                "Consider using a GPU-enabled environment for large-scale evaluation."
            )
    
    def _make_cache_key(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        """Create a deterministic cache key from request parameters."""
        request_dict = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        request_str = json.dumps(request_dict, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> str:
        """
        Generate a completion for chat-formatted messages.
        
        Args:
            messages: List of dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 = greedy)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text string
        """
        # Convert chat format to prompt using Llama template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode only the generated tokens (skip input)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat_completion_with_logprobs(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> tuple[str, float]:
        """
        Generate completion and return with log probability.
        
        Args:
            messages: List of dicts with 'role' and 'content' keys
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            (generated_text, log_probability)
        """
        # Check cache first (only for deterministic generation, not sampling)
        cache_key = None
        sampling_mode = float(temperature) > 0.0
        if self.cache is not None and not sampling_mode:
            cache_key = self._make_cache_key(messages, temperature, max_tokens)
            hit = self.cache.get(cache_key)
            if hit.hit and hit.content is not None:
                # Return cached result
                logprob = hit.token_logprobs[0] if hit.token_logprobs else 0.0
                return hit.content, logprob
        
        # Convert chat format to prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,  # Get logits
        }
        
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode generated tokens
        generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        # Compute log probability
        # outputs.scores is a tuple of tensors, one per generated token
        # Each tensor has shape (batch_size, vocab_size)
        log_prob = 0.0
        if hasattr(outputs, 'scores') and outputs.scores:
            import torch.nn.functional as F
            for i, score_tensor in enumerate(outputs.scores):
                if i >= len(generated_ids):
                    break
                # Get log probabilities
                log_probs = F.log_softmax(score_tensor[0], dim=-1)
                # Get the log prob of the actual generated token
                token_id = generated_ids[i]
                log_prob += log_probs[token_id].item()
        
        # Store in cache (only for deterministic generation, not sampling)
        if self.cache is not None and cache_key is not None and not sampling_mode:
            self.cache.put(
                cache_key,
                request={"model": self.model_name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                response={"text": response.strip(), "logprob": log_prob},
                content=response.strip(),
                token_logprobs=[log_prob],
                usage_prompt_tokens=len(inputs.input_ids[0]),
                usage_completion_tokens=len(generated_ids),
                duration_ms=0,  # Will be updated in calibration.py
            )
        
        return response.strip(), log_prob
    
    def supports_logprobs(self) -> bool:
        """This client provides token logprobs via direct model access."""
        return True

