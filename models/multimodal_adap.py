import torch
import torch.nn as nn
import logging
from typing import Dict, Any
from transformers import LlamaForCausalLM

from models.adapters import MultiModalAdapter
from utils.config import MMTConfig

logger = logging.getLogger("trainer")


class MultiModalAdap(nn.Module):
    """Multimodal Adap model.

    Builds a fixed-length multimodal prefix via `MultiModalAdapter`, prepends it to token
    embeddings, runs the base LLaMA, then scores KG candidates with a linear head.
    """

    def __init__(
        self,
        model: LlamaForCausalLM,
        num_prefix: int = 1,
        kge_model_path: str = None,
        adapter_config: Dict[str, Any] = None,
    ):
        super().__init__()

        self.llama_model = model
        if hasattr(model, "llama_model"):
            self.hidden_size = model.llama_model.config.hidden_size
        else:
            self.hidden_size = model.config.hidden_size

        self.num_prefix = num_prefix
        self.device = MMTConfig.get_device("train")

        self.multimodal_adapter = MultiModalAdapter(
            kge_model_path=kge_model_path,
            adapter_config=adapter_config,
        ).to(self.device)

        self.kg_score_head = nn.Linear(self.hidden_size, 1)
        nn.init.normal_(self.kg_score_head.weight, std=0.1)
        nn.init.zeros_(self.kg_score_head.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        embedding_ids=None,      # [B, 3] (h, r, t)
        kg_candidate_ids=None,   # [B, 1+K] candidate entity ids
        entity_types=None,       # [B, 2] optional (head_type, tail_type)
        use_cache: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        """Forward pass returning LM loss/logits and optional KG scores."""

        kg_scores = None
        prefix_disabled = False

        if embedding_ids is not None:
            if embedding_ids.dim() == 1:
                embedding_ids = embedding_ids.unsqueeze(0)

            batch_size = embedding_ids.size(0)

            token_embeds = self.llama_model.get_input_embeddings()(input_ids)
            if token_embeds.dim() == 2:
                token_embeds = token_embeds.unsqueeze(0)
            
            target_dtype = torch.float32
            
            if token_embeds.dtype != target_dtype:
                logger.warning(
                    f"Unexpected token_embeds dtype: {token_embeds.dtype}, converting to {target_dtype}"
                )
                token_embeds = token_embeds.to(dtype=target_dtype)
            
            prefix_embeds = self.multimodal_adapter(
                embedding_ids, 
                entity_types=entity_types,
                target_dtype=target_dtype
            )  # [B, P, H]

            if prefix_embeds.dtype != token_embeds.dtype:
                logger.warning(
                    f"prefix_embeds dtype ({prefix_embeds.dtype}) != token_embeds dtype ({token_embeds.dtype}), "
                    f"forcing conversion to float32. This should not happen with unified float32 config."
                )
                prefix_embeds = prefix_embeds.to(dtype=torch.float32)
                token_embeds = token_embeds.to(dtype=torch.float32)
            
            prefix_std = prefix_embeds.std().item()
            prefix_max_abs = prefix_embeds.abs().max().item()
            
            prefix_disabled = False
            
            if prefix_max_abs < 1e-3:
                logger.warning(
                    f"prefix_embeds is constant/zero (std={prefix_std:.6f}, max_abs={prefix_max_abs:.6f}), "
                    f"DISABLING prefix and using pure token_embeds to avoid LLaMA overflow."
                )
                inputs_embeds = token_embeds
                prefix_disabled = True
            else:
                if torch.isnan(prefix_embeds).any() or torch.isinf(prefix_embeds).any():
                    nan_count = torch.isnan(prefix_embeds).sum().item()
                    inf_count = torch.isinf(prefix_embeds).sum().item()
                    logger.warning(
                        f"prefix_embeds contains NaN/Inf before normalization (NaN: {nan_count}, Inf: {inf_count}), "
                        f"replacing with zeros."
                    )
                    
                
                
            
                PREFIX_SCALE = 0.2
                prefix_embeds = prefix_embeds * PREFIX_SCALE
                prefix_embeds = torch.clamp(prefix_embeds, -2.0, 2.0)
                
                if torch.isnan(prefix_embeds).any() or torch.isinf(prefix_embeds).any():
                    nan_count = torch.isnan(prefix_embeds).sum().item()
                    inf_count = torch.isinf(prefix_embeds).sum().item()
                    logger.error(
                        f"CRITICAL: prefix_embeds still contains NaN/Inf after normalization (NaN: {nan_count}, Inf: {inf_count}). "
                        f"DISABLING prefix and using pure token_embeds."
                    )
                    inputs_embeds = token_embeds
                    prefix_disabled = True
                else:
                    inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
                    
                    if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
                        nan_count = torch.isnan(inputs_embeds).sum().item()
                        inf_count = torch.isinf(inputs_embeds).sum().item()
                        logger.error(
                            f"CRITICAL: inputs_embeds contains NaN/Inf before LLaMA forward (NaN: {nan_count}, Inf: {inf_count}). "
                            f"DISABLING prefix and using pure token_embeds."
                        )
                        inputs_embeds = token_embeds
                        prefix_disabled = True

            if not prefix_disabled and inputs_embeds.shape[1] > token_embeds.shape[1]:
                actual_prefix_len = inputs_embeds.shape[1] - token_embeds.shape[1]
                
                if attention_mask is not None:
                    if attention_mask.dim() == 1:
                        attention_mask = attention_mask.unsqueeze(0)
                    prefix_mask = torch.ones(
                        (batch_size, actual_prefix_len),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

                if labels is not None:
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(0)
                    ignore_pad = torch.full(
                        (batch_size, actual_prefix_len),
                        -100,
                        device=labels.device,
                        dtype=labels.dtype,
                    )
                    labels = torch.cat([ignore_pad, labels], dim=1)
        else:
            inputs_embeds = None

        if inputs_embeds is not None:
            if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
                logger.error("CRITICAL: inputs_embeds contains NaN/Inf right before LLaMA forward!")
                
            
            outputs = self.llama_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True,
            )
            
            if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                last_hidden = outputs.hidden_states[-1]
                if torch.isnan(last_hidden).any() or torch.isinf(last_hidden).any():
                    logger.error(
                        "CRITICAL: LLaMA hidden_states contains NaN/Inf. Abort step to prevent gradient corruption."
                    )
                    raise RuntimeError(
                        "NaN/Inf detected in LLaMA hidden_states. Abort step to prevent gradient corruption."
                    )
        else:
            outputs = self.llama_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True,
            )

        if embedding_ids is not None and kg_candidate_ids is not None:
            hidden_states = outputs.hidden_states[-1]

            if prefix_disabled:
                prefix_end_idx = 0
                logger.warning(
                    "prefix is disabled, using first token position for kg_ctx extraction."
                )
            elif inputs_embeds is not None and inputs_embeds.shape[1] > token_embeds.shape[1]:
                actual_prefix_len = inputs_embeds.shape[1] - token_embeds.shape[1]
                prefix_end_idx = actual_prefix_len - 1
            else:
                prefix_end_idx = 0

            kg_ctx = hidden_states[:, prefix_end_idx, :]  # [B, H]
            
            if torch.isnan(kg_ctx).any() or torch.isinf(kg_ctx).any():
                nan_count = torch.isnan(kg_ctx).sum().item()
                inf_count = torch.isinf(kg_ctx).sum().item()
                logger.warning(
                    f"kg_ctx contains NaN/Inf (NaN: {nan_count}, Inf: {inf_count}), "
                    f"replacing with zeros. This may indicate gradient explosion or numerical instability."
                )
                
            
            

            if kg_candidate_ids.dim() == 1:
                kg_candidate_ids = kg_candidate_ids.unsqueeze(0)  # [N] -> [1, N]
            elif kg_candidate_ids.dim() == 3:
                if kg_candidate_ids.size(1) == 1:
                    kg_candidate_ids = kg_candidate_ids.squeeze(1)
                elif kg_candidate_ids.size(2) == 1:
                    kg_candidate_ids = kg_candidate_ids.squeeze(2)

            B, N = kg_candidate_ids.shape  # [B, N]

            kg_ctx = kg_ctx.unsqueeze(1).expand(-1, N, -1)  # [B, N, H]

            rel_ids = None
            if embedding_ids is not None:
                rel_ids = embedding_ids[:, 1]  # [B]

            if rel_ids is not None:
                rel_ids_for_candidates = rel_ids.unsqueeze(1).expand(-1, N)  # [B, N]
            else:
                rel_ids_for_candidates = None

            tail_embeds_dtype = kg_ctx.dtype  # match model compute dtype
            tail_embeds = self.multimodal_adapter.get_entity_embedding(
                kg_candidate_ids,
                proj=True,
                target_dtype=tail_embeds_dtype,
                rel_ids=rel_ids_for_candidates,
            )  # [B, N, H]
            
            if torch.isnan(tail_embeds).any() or torch.isinf(tail_embeds).any():
                nan_count = torch.isnan(tail_embeds).sum().item()
                inf_count = torch.isinf(tail_embeds).sum().item()
                logger.warning(
                    f"tail_embeds contains NaN/Inf (NaN: {nan_count}, Inf: {inf_count}), "
                    f"replacing with zeros. This may indicate adapter output issue."
                )
                
            
            combined = kg_ctx * tail_embeds  # [B, N, H]
            
            if torch.isnan(combined).any() or torch.isinf(combined).any():
                nan_count = torch.isnan(combined).sum().item()
                inf_count = torch.isinf(combined).sum().item()
                logger.warning(
                    f"combined features contain NaN/Inf (NaN: {nan_count}, Inf: {inf_count}), "
                    f"replacing with zeros."
                )
                
            
            

            kg_scores = self.kg_score_head(combined).squeeze(-1)  # [B, N]
            
            if torch.isnan(kg_scores).any() or torch.isinf(kg_scores).any():
                nan_count = torch.isnan(kg_scores).sum().item()
                inf_count = torch.isinf(kg_scores).sum().item()
                logger.warning(
                    f"kg_scores from score head contain NaN/Inf (NaN: {nan_count}, Inf: {inf_count}), "
                    f"replacing with zeros."
                )
                
            
            

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "kg_scores": kg_scores,
        }

    def generate(self, **kwargs):
        """Proxy to the underlying LLaMA generate()."""
        return self.llama_model.generate(**kwargs)

    def save_adapters(self, output_dir: str):
        """Save adapter weights."""
        torch.save(
            self.multimodal_adapter.state_dict(),
            f"{output_dir}/multimodal_adapters.pth",
        )

    def load_adapters(self, adapter_path: str):
        """Load adapter weights (supports legacy 'multimodal_adapter.' prefix)."""
        adapter_weights = torch.load(adapter_path, map_location="cpu")

        fixed_weights = {}
        for key, value in adapter_weights.items():
            if key.startswith("multimodal_adapter."):
                new_key = key.replace("multimodal_adapter.", "")
                fixed_weights[new_key] = value
            else:
                fixed_weights[key] = value

        self.multimodal_adapter.load_state_dict(fixed_weights)
        print(f"Adapter weights loaded: {adapter_path}")


