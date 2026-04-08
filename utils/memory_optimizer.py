import torch
import logging
from typing import Dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """LoRA-only memory optimizer (no quantization; float32 by default)."""

    def __init__(self, strategy: str = "lora"):
        self.strategy = strategy
        self.strategy_configs = {
            "lora": {
                # LoRA config (no quantization)
                "torch_dtype": torch.float32,

                # LoRA params
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                ],

                # Training stability
                "gradient_checkpointing": True,
            }
        }

        if strategy not in self.strategy_configs:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategy_configs.keys())}")

    def get_optimized_model(
        self,
        base_model_name_or_path: str,
    ):
        """
        Returns:
            model: LoRA-wrapped model (float32, no quantization)
            tokenizer
        """
        cfg = self.strategy_configs[self.strategy]

        logger.info(f"Loading base model with LoRA (strategy: {self.strategy})")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            device_map="auto",
            torch_dtype=cfg["torch_dtype"],
            use_cache=False,
        )

        logger.info(f"Base model loaded (dtype: {cfg['torch_dtype']})")

        if cfg.get("gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
            logger.info("Gradient checkpointing enabled")

        lora_config = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            target_modules=cfg["lora_target_modules"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

        model.config.use_cache = False

        trainable, total = self._count_params(model)
        logger.info(
            f"LoRA attached | Trainable params: {trainable / 1e6:.2f}M "
            f"/ Total: {total / 1e6:.2f}M"
        )
        logger.info(f"Model dtype: {next(model.parameters()).dtype}")

        return model, tokenizer

    def _count_params(self, model):
        trainable = 0
        total = 0
        for p in model.parameters():
            num = p.numel()
            total += num
            if p.requires_grad:
                trainable += num
        return trainable, total
