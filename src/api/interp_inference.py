"""NNsight-based LLM interface for interpretability experiments."""

import logging
from typing import Callable, Dict, List

import torch
from nnsight import LanguageModel, save

from models.api import LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


class InterpInference:
    """LLM interface using NNsight for inference
    with interpretability support."""

    def __init__(
        self,
        model: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
        **model_kwargs,
    ):
        """Initialize the NNsight language model.

        Args:
            model: HuggingFace model identifier
            (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
            device: Device to run on ("cpu", "cuda", "cuda:0", etc.)
            dtype: Torch dtype for model weights
            **model_kwargs: Additional kwargs passed to LanguageModel
        """
        logger.info(f"Loading NNsight model: {model} on {device}")

        self.model = LanguageModel(
            model,
            device_map=device if device != "cpu" else None,
            dtype=dtype,
            **model_kwargs,
        )

        logger.info(f"Model {model} loaded successfully")

        self.device = device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        input_len: int = None,
        intervention: Callable = None,
        **generate_kwargs,
    ) -> str:
        """Generate text from a single prompt."""
        logger.debug(f"Generating with prompt: {prompt[:50]}...")

        out = None
        with self.model.generate(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **generate_kwargs,
        ) as tracer:
            intervention_output = save(list())

            with tracer.invoke(prompt):
                with tracer.iter[:]:
                    intervention_output.append(intervention(self.model))

            with tracer.invoke():
                out = save(self.model.generator.output)

        if out is not None:
            generated_tokens = out[0, input_len:]
            decoded_answer = self.model.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
        else:
            decoded_answer = ""

        return decoded_answer, intervention_output

    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        intervention: Callable = None,
        **generate_kwargs,
    ) -> LLMResponse:
        """Generate response from chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **generate_kwargs: Additional generation kwargs

        Returns:
            LLMResponse with content and token counts
        """
        # Apply chat template if available
        if hasattr(self.model.tokenizer, "apply_chat_template"):
            prompt = self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: concatenate messages
            prompt = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in messages
            )

        # Tokenize to get input length
        input_ids = self.model.tokenizer(prompt, return_tensors="pt")
        input_token_count = input_ids["input_ids"].shape[1]

        # Generate
        output, intervention_output = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            input_len=input_token_count,
            intervention=intervention,
            **generate_kwargs,
        )

        # Estimate output tokens (full output minus input)
        output_ids = self.model.tokenizer(output, return_tensors="pt")
        output_token_count = max(
            1, output_ids["input_ids"].shape[1] - input_token_count
        )

        # get the name of the intervention
        intervention_name = (
            intervention.__name__ if intervention else "no_intervention"
        )
        intervention_output = {intervention_name: intervention_output}

        return LLMResponse(
            content=output,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            intervention_output=intervention_output,
        )

    @property
    def tokenizer(self):
        """Access the underlying tokenizer."""
        return self.model.tokenizer
