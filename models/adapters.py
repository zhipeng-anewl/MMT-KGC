"""Multi-modal adapter used by MultiModalAdap.

Implements relation-guided fusion (inspired by NativE):
1) Load unified multimodal features (visual, textual, numeric)
2) Project all modalities to unified dimension
3) Relation-guided attention fusion (per entity, per relation)
4) Output prefix: [h_joint, r, t_joint] (3 tokens) instead of 7 tokens
"""

from typing import Dict, Optional, Tuple
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import MMTConfig


class MultiModalAdapter(nn.Module):
    def __init__(
            self,
        kge_model_path: Optional[str] = None,
        adapter_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        # Adapter hyper-params (override via training config if needed)
        default_cfg = {
            "visual_dim": 2048,
            "textual_dim": 768,
            "numeric_dim": 7,
            "kge_ent_dim": 128,
            "kge_rel_dim": 64,
            "fusion_dim": 128,
            "llm_dim": 2048,
            "num_prefix": 1,
            # modality ablations
            "use_visual": True,
            "use_textual": True,
            "use_numeric": True,
            # relation-guided fusion
            "use_relation_attention": True,
            "relation_fusion_type": "mean",  # used when use_relation_attention=False ("mean" or "mlp")
        }
        if adapter_config:
            default_cfg.update(adapter_config)
        self.cfg = default_cfg
        self.device = MMTConfig.get_device("train")

        # Pretrained KGE
        ent_embs, rel_embs = self._load_pretrain_kge(kge_model_path)
        self.register_buffer("kge_ent_embs", ent_embs)
        self.register_buffer("kge_rel_embs", rel_embs)

        # Project each modality to fusion_dim
        self.visual_proj = nn.Sequential(
            nn.Linear(self.cfg["visual_dim"], self.cfg["fusion_dim"]),
            nn.ReLU(),
            nn.Linear(self.cfg["fusion_dim"], self.cfg["fusion_dim"])
        )
        self.textual_proj = nn.Sequential(
            nn.Linear(self.cfg["textual_dim"], self.cfg["fusion_dim"]),
            nn.ReLU(),
            nn.Linear(self.cfg["fusion_dim"], self.cfg["fusion_dim"])
        )
        self.numeric_proj = nn.Sequential(
            nn.Linear(self.cfg["numeric_dim"], self.cfg["fusion_dim"]),
            nn.ReLU(),
            nn.Linear(self.cfg["fusion_dim"], self.cfg["fusion_dim"])
        )

        # Relation-guided attention fusion
        self.ent_attn = nn.Linear(self.cfg["fusion_dim"], 1, bias=False)

        # Relation gate: controls attention temperature per relation id
        with open(MMTConfig.FILE_PATHS["relation2id"], 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines and len(lines[0].split()) == 1 and lines[0].isdigit():
            nrelation = int(lines[0])
        else:
            nrelation = len(lines)
        self.rel_gate = nn.Embedding(nrelation, 1)

        # Optional baseline fusion when relation attention is disabled
        self.relation_fusion_mlp = nn.Sequential(
            nn.Linear(self.cfg["fusion_dim"] * 4, self.cfg["fusion_dim"]),
            nn.ReLU(),
            nn.Linear(self.cfg["fusion_dim"], self.cfg["fusion_dim"]),
        )
        
        # Project fused features to LLM hidden size
        self.fusion_to_llm = nn.Linear(
            self.cfg["fusion_dim"], 
            self.cfg["llm_dim"] * self.cfg["num_prefix"]
        )
        self.rel_adapter = nn.Linear(
            self.cfg["kge_rel_dim"], 
            self.cfg["llm_dim"] * self.cfg["num_prefix"]
        )
        self.prefix_norm = nn.LayerNorm(self.cfg["llm_dim"], eps=1e-6)

        self._reset_parameters()
        self._load_multimodal_data()

        self.to(self.device)

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #
    def _load_pretrain_kge(self, path: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load pretrained RotatE entity/relation embeddings."""
        try:
            ent_embs = np.load(MMTConfig.FILE_PATHS["kge_entity"])
            rel_embs = np.load(MMTConfig.FILE_PATHS["kge_relation"])
        except Exception:
            # Fallback to small random tensors to keep training runnable
            ent_embs = np.random.randn(18438, self.cfg["kge_ent_dim"]) * 0.1
            rel_embs = np.random.randn(6, self.cfg["kge_rel_dim"]) * 0.1

        def _clean(arr: np.ndarray, clamp: float = 2.0) -> np.ndarray:
            arr = np.nan_to_num(arr, nan=0.0, posinf=clamp, neginf=-clamp)
            return np.clip(arr, -clamp, clamp)

        ent_embs = _clean(ent_embs)
        rel_embs = _clean(rel_embs)

        ent_t = torch.tensor(ent_embs, dtype=torch.float32)
        rel_t = torch.tensor(rel_embs, dtype=torch.float32)
        return ent_t.to(self.device), rel_t.to(self.device)

    def _load_multimodal_data(self) -> None:
        """Load unified multimodal features (visual, textual, numeric)."""
        def _clean_tensor(t: torch.Tensor, clamp: float = 2.0) -> torch.Tensor:
            t = torch.nan_to_num(t, nan=0.0, posinf=clamp, neginf=-clamp)
            return torch.clamp(t, -clamp, clamp)

        try:
            if os.path.exists(MMTConfig.FILE_PATHS["visual"]):
                self.visual_embs = _clean_tensor(torch.load(MMTConfig.FILE_PATHS["visual"])).to(self.device)
            else:
                print(f"Visual feature file not found: {MMTConfig.FILE_PATHS['visual']}")
                total_entities = self.kge_ent_embs.shape[0]
                self.visual_embs = torch.zeros(total_entities, self.cfg["visual_dim"], device=self.device)
            
            if os.path.exists(MMTConfig.FILE_PATHS["textual"]):
                self.textual_embs = _clean_tensor(torch.load(MMTConfig.FILE_PATHS["textual"])).to(self.device)
            else:
                print(f"Textual feature file not found: {MMTConfig.FILE_PATHS['textual']}")
                total_entities = self.kge_ent_embs.shape[0]
                self.textual_embs = torch.zeros(total_entities, self.cfg["textual_dim"], device=self.device)
            
            if os.path.exists(MMTConfig.FILE_PATHS["numeric"]):
                self.numeric_embs = _clean_tensor(torch.load(MMTConfig.FILE_PATHS["numeric"])).to(self.device)
            else:
                print(f"Numeric feature file not found: {MMTConfig.FILE_PATHS['numeric']}")
                total_entities = self.kge_ent_embs.shape[0]
                self.numeric_embs = torch.zeros(total_entities, self.cfg["numeric_dim"], device=self.device)
            
            # Register as buffers (no gradients)
            if not hasattr(self, "visual_embs"):
                self.register_buffer("visual_embs", self.visual_embs)
            else:
                self.visual_embs.data = self.visual_embs.to(self.device)
            
            if not hasattr(self, "textual_embs"):
                self.register_buffer("textual_embs", self.textual_embs)
            else:
                self.textual_embs.data = self.textual_embs.to(self.device)
            
            if not hasattr(self, "numeric_embs"):
                self.register_buffer("numeric_embs", self.numeric_embs)
            else:
                self.numeric_embs.data = self.numeric_embs.to(self.device)
        except Exception as e:
            print(f"Failed to load multimodal features: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to zeros
            total_entities = self.kge_ent_embs.shape[0]
            visual_embs = torch.zeros(total_entities, self.cfg["visual_dim"], device=self.device)
            textual_embs = torch.zeros(total_entities, self.cfg["textual_dim"], device=self.device)
            numeric_embs = torch.zeros(total_entities, self.cfg["numeric_dim"], device=self.device)
            if not hasattr(self, "visual_embs"):
                self.register_buffer("visual_embs", visual_embs)
            else:
                self.visual_embs.data = visual_embs
            
            if not hasattr(self, "textual_embs"):
                self.register_buffer("textual_embs", textual_embs)
            else:
                self.textual_embs.data = textual_embs
            
            if not hasattr(self, "numeric_embs"):
                self.register_buffer("numeric_embs", numeric_embs)
            else:
                self.numeric_embs.data = numeric_embs

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        # Projectors
        for layer in [self.visual_proj, self.textual_proj, self.numeric_proj]:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    nn.init.zeros_(sublayer.bias)

        # Attention
        nn.init.xavier_uniform_(self.ent_attn.weight)

        # Relation gate
        nn.init.uniform_(self.rel_gate.weight, -0.1, 0.1)

        # MLP baseline
        for sublayer in self.relation_fusion_mlp:
            if isinstance(sublayer, nn.Linear):
                nn.init.xavier_uniform_(sublayer.weight)
                nn.init.zeros_(sublayer.bias)

        # Fused -> LLM
        nn.init.xavier_uniform_(self.fusion_to_llm.weight)
        nn.init.zeros_(self.fusion_to_llm.bias)

        # Relation -> LLM
        nn.init.xavier_uniform_(self.rel_adapter.weight)
        nn.init.zeros_(self.rel_adapter.bias)

    # ------------------------------------------------------------------ #
    # Core building blocks: Relation-guided Fusion
    # ------------------------------------------------------------------ #
    def _get_joint_embeddings(
        self, 
        struct_emb: torch.Tensor,   # [B, fusion_dim]
        visual_emb: torch.Tensor,   # [B, fusion_dim]
        textual_emb: torch.Tensor,  # [B, fusion_dim]
        numeric_emb: torch.Tensor,  # [B, fusion_dim]
        rel_gate: torch.Tensor      # [B, 1]
    ) -> torch.Tensor:
        """
        Relation-guided attention fusion (inspired by NativE)
        
        Args:
            struct_emb: structural embeddings [B, fusion_dim]
            visual_emb: visual embeddings [B, fusion_dim]
            textual_emb: textual embeddings [B, fusion_dim]
            numeric_emb: numeric embeddings [B, fusion_dim]
            rel_gate: relation gate [B, 1]
        
        Returns:
            joint_emb: fused embeddings [B, fusion_dim]
        """
        # Stack modalities: [B, 4, fusion_dim]
        e = torch.stack([struct_emb, visual_emb, textual_emb, numeric_emb], dim=1)

        # Baseline fusion (no relation-guided attention)
        if not self.cfg.get("use_relation_attention", True):
            fusion_type = self.cfg.get("relation_fusion_type", "mean")

            if fusion_type == "mlp":
                e_flat = e.view(e.size(0), -1)
                joint_emb = self.relation_fusion_mlp(e_flat)  # [B, fusion_dim]
                return joint_emb
            else:
                joint_emb = e.mean(dim=1)  # [B, fusion_dim]
                return joint_emb

        # Relation-guided attention fusion
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)

        attention_weights = torch.softmax(scores / (torch.sigmoid(rel_gate) + 1e-8), dim=-1)

        joint_emb = torch.sum(attention_weights.unsqueeze(-1) * e, dim=1)
        
        return joint_emb

    @torch.no_grad()
    def get_relation_guided_modality_attention(
        self,
        entity_ids: torch.Tensor,
        rel_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute relation-guided fusion modality attention weights for tail-only case study.

        Attention weights are produced by MultiModalAdapter._get_joint_embeddings:
            attention_weights: softmax(scores / sigmoid(rel_gate), dim=-1)
        where modalities order is:
            [0]=struct, [1]=visual, [2]=textual, [3]=numeric

        Args:
            entity_ids: Tensor of entity ids, any shape [...]. Typically tail entity ids.
            rel_ids: Tensor of relation ids. Must be broadcastable to entity_ids shape.

        Returns:
            attention_weights: Tensor of shape [..., 4]
        """
        if rel_ids is None:
            raise ValueError("rel_ids cannot be None for relation-guided attention computation.")

        entity_ids = entity_ids.to(self.device)
        rel_ids = rel_ids.to(self.device)

        # Make rel_ids shape compatible with entity_ids
        if rel_ids.numel() == 1 and entity_ids.numel() > 1:
            rel_ids = rel_ids.expand_as(entity_ids)
        elif rel_ids.shape != entity_ids.shape:
            raise ValueError(
                f"rel_ids shape must equal entity_ids shape (or be scalar). "
                f"Got entity_ids.shape={tuple(entity_ids.shape)}, rel_ids.shape={tuple(rel_ids.shape)}"
            )

        # Clamp structure embeddings to match the adapter forward path
        struct_emb = torch.clamp(self.kge_ent_embs[entity_ids], -2.0, 2.0)  # [..., kge_ent_dim]
        visual_emb = self.visual_proj(self.visual_embs[entity_ids])  # [..., fusion_dim]
        textual_emb = self.textual_proj(self.textual_embs[entity_ids])  # [..., fusion_dim]
        numeric_emb = self.numeric_proj(self.numeric_embs[entity_ids])  # [..., fusion_dim]

        # Multi-modal ablations (match get_entity_embedding)
        if not self.cfg.get("use_visual", True):
            visual_emb = torch.zeros_like(visual_emb)
        if not self.cfg.get("use_textual", True):
            textual_emb = torch.zeros_like(textual_emb)
        if not self.cfg.get("use_numeric", True):
            numeric_emb = torch.zeros_like(numeric_emb)

        # Project struct embeddings to fusion_dim if needed
        if struct_emb.size(-1) != self.cfg["fusion_dim"]:
            if not hasattr(self, "struct_to_fusion"):
                # This layer is only needed when kge_ent_dim != fusion_dim
                self.struct_to_fusion = nn.Linear(self.cfg["kge_ent_dim"], self.cfg["fusion_dim"]).to(self.device)
                nn.init.xavier_uniform_(self.struct_to_fusion.weight)
                nn.init.zeros_(self.struct_to_fusion.bias)
            struct_emb = self.struct_to_fusion(struct_emb)

        original_shape = struct_emb.shape[:-1]  # [...]
        fusion_dim = struct_emb.shape[-1]

        # Flatten to [B, fusion_dim] for reuse of _get_joint_embeddings math
        struct_flat = struct_emb.view(-1, fusion_dim)
        visual_flat = visual_emb.view(-1, fusion_dim)
        textual_flat = textual_emb.view(-1, fusion_dim)
        numeric_flat = numeric_emb.view(-1, fusion_dim)
        rel_gate_flat = self.rel_gate(rel_ids).view(-1, 1)  # [B, 1]

        e = torch.stack([struct_flat, visual_flat, textual_flat, numeric_flat], dim=1)  # [B, 4, fusion_dim]
        u = torch.tanh(e)
        scores = self.ent_attn(u).squeeze(-1)  # [B, 4]
        attention_weights = torch.softmax(scores / (torch.sigmoid(rel_gate_flat) + 1e-8), dim=-1)  # [B, 4]

        return attention_weights.view(*original_shape, 4)

    def _infer_entity_type_by_id(self, entity_id: int) -> str:
        if 0 <= entity_id <= 866:
            return "disease"
        if 867 <= entity_id <= 1368:
            return "herb"
        if 1369 <= entity_id <= 15097:
            return "molecule"
        if 15098 <= entity_id <= 18437:
            return "target"
        return "disease"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def forward(
        self,
        triple_ids: torch.Tensor,
        entity_types: Optional[torch.Tensor] = None,
        target_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Generate prefix using relation-guided fusion:
        [h_joint, r, t_joint] => length 3 * num_prefix when num_prefix=1.
        
        Process:
        1. Load multimodal features for head and tail entities
        2. Project all modalities to fusion_dim
        3. Relation-guided attention fusion (per entity, per relation)
        4. Project fused embeddings to LLM space
        5. Concatenate: [h_joint, r, t_joint]
        """
        triple_ids = triple_ids.to(self.device)
        batch_size = triple_ids.size(0)
        head_ids = triple_ids[:, 0]  # [B]
        rel_ids = triple_ids[:, 1]   # [B]
        tail_ids = triple_ids[:, 2]  # [B]
        
        # Prevent data leakage: treat tail_id==0 as a placeholder and use zeros for tail features.
        is_placeholder = (tail_ids == 0)  # [B] bool

        # 1) Structural embeddings
        head_struct = torch.clamp(self.kge_ent_embs[head_ids], -2.0, 2.0)  # [B, kge_ent_dim]
        tail_struct = torch.where(
            is_placeholder.unsqueeze(-1),
            torch.zeros_like(head_struct),
            torch.clamp(self.kge_ent_embs[tail_ids], -2.0, 2.0),
        )  # [B, kge_ent_dim]
        
        # Project to fusion_dim if needed
        if head_struct.size(-1) != self.cfg["fusion_dim"]:
            if not hasattr(self, 'struct_to_fusion'):
                self.struct_to_fusion = nn.Linear(self.cfg["kge_ent_dim"], self.cfg["fusion_dim"]).to(self.device)
                nn.init.xavier_uniform_(self.struct_to_fusion.weight)
                nn.init.zeros_(self.struct_to_fusion.bias)
            head_struct = self.struct_to_fusion(head_struct)
            tail_struct = self.struct_to_fusion(tail_struct)
            
        # 2) Multimodal features -> fusion_dim
        head_visual = self.visual_proj(self.visual_embs[head_ids])  # [B, fusion_dim]
        head_textual = self.textual_proj(self.textual_embs[head_ids])  # [B, fusion_dim]
        head_numeric = self.numeric_proj(self.numeric_embs[head_ids])  # [B, fusion_dim]
        
        tail_visual_base = torch.where(
            is_placeholder.unsqueeze(-1),
            torch.zeros_like(self.visual_embs[head_ids]),
            self.visual_embs[tail_ids],
        )
        tail_visual = self.visual_proj(tail_visual_base)  # [B, fusion_dim]
        
        tail_textual_base = torch.where(
            is_placeholder.unsqueeze(-1),
            torch.zeros_like(self.textual_embs[head_ids]),
            self.textual_embs[tail_ids],
        )
        tail_textual = self.textual_proj(tail_textual_base)  # [B, fusion_dim]
        
        tail_numeric_base = torch.where(
            is_placeholder.unsqueeze(-1),
            torch.zeros_like(self.numeric_embs[head_ids]),
            self.numeric_embs[tail_ids],
        )
        tail_numeric = self.numeric_proj(tail_numeric_base)  # [B, fusion_dim]

        # Modality ablations
        if not self.cfg.get("use_visual", True):
            head_visual = torch.zeros_like(head_visual)
            tail_visual = torch.zeros_like(tail_visual)
        if not self.cfg.get("use_textual", True):
            head_textual = torch.zeros_like(head_textual)
            tail_textual = torch.zeros_like(tail_textual)
        if not self.cfg.get("use_numeric", True):
            head_numeric = torch.zeros_like(head_numeric)
            tail_numeric = torch.zeros_like(tail_numeric)

        # 3) Relation-guided fusion
        rel_gate = self.rel_gate(rel_ids)  # [B, 1]

        h_joint = self._get_joint_embeddings(
            struct_emb=head_struct,
            visual_emb=head_visual,
            textual_emb=head_textual,
            numeric_emb=head_numeric,
            rel_gate=rel_gate
        )  # [B, fusion_dim]
        
        t_joint = self._get_joint_embeddings(
            struct_emb=tail_struct,
            visual_emb=tail_visual,
            textual_emb=tail_textual,
            numeric_emb=tail_numeric,
            rel_gate=rel_gate
        )  # [B, fusion_dim]
        
        # 4) Project to LLM space
        h_joint_llm = self.fusion_to_llm(h_joint).view(batch_size, self.cfg["num_prefix"], self.cfg["llm_dim"])  # [B, num_prefix, llm_dim]
        t_joint_llm = self.fusion_to_llm(t_joint).view(batch_size, self.cfg["num_prefix"], self.cfg["llm_dim"])  # [B, num_prefix, llm_dim]
        
        rel_emb = torch.clamp(self.kge_rel_embs[rel_ids], -2.0, 2.0)  # [B, kge_rel_dim]
        r_llm = self.rel_adapter(rel_emb).view(batch_size, self.cfg["num_prefix"], self.cfg["llm_dim"])  # [B, num_prefix, llm_dim]

        # 5) Concatenate prefix: [h_joint, r, t_joint]
        prefix = torch.cat([h_joint_llm, r_llm, t_joint_llm], dim=1)  # [B, 3 * num_prefix, llm_dim]

        # ===== 6. Normalize + clamp for stability =====
        prefix = self.prefix_norm(prefix)
        prefix = torch.clamp(prefix, -5.0, 5.0)

        # Default to float32 unless target_dtype is specified
        if target_dtype is not None:
            prefix = prefix.to(dtype=target_dtype)
        else:
            prefix = prefix.to(dtype=torch.float32)

        return prefix

    def get_entity_embedding(
        self, 
        entity_ids: torch.Tensor, 
        proj: bool = True, 
        target_dtype: Optional[torch.dtype] = None,
        rel_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return entity embeddings (projected to LLM space when proj=True).
        
        Args:
            entity_ids: entity ids, any shape [B, ...]
            proj: whether to project to LLM space
            target_dtype: output dtype
            rel_ids: optional relation ids (enables relation-guided fusion)
        
        Returns:
            entity_emb: [B, ..., llm_dim]
        """
        entity_ids = entity_ids.to(self.device)
        struct_emb = torch.clamp(self.kge_ent_embs[entity_ids], -2.0, 2.0)  # [B, ..., kge_ent_dim]

        if not proj:
            return struct_emb

        # Relation-guided fusion if rel_ids is provided
        if rel_ids is not None:
            rel_ids = rel_ids.to(self.device)
            rel_gate = self.rel_gate(rel_ids)  # [B, ..., 1]
            
            # Multimodal features -> fusion_dim
            visual_emb = self.visual_proj(self.visual_embs[entity_ids])  # [B, ..., fusion_dim]
            textual_emb = self.textual_proj(self.textual_embs[entity_ids])  # [B, ..., fusion_dim]
            numeric_emb = self.numeric_proj(self.numeric_embs[entity_ids])  # [B, ..., fusion_dim]

            # Modality ablations
            if not self.cfg.get("use_visual", True):
                visual_emb = torch.zeros_like(visual_emb)
            if not self.cfg.get("use_textual", True):
                textual_emb = torch.zeros_like(textual_emb)
            if not self.cfg.get("use_numeric", True):
                numeric_emb = torch.zeros_like(numeric_emb)
            
            # Project struct embeddings to fusion_dim
            if struct_emb.size(-1) != self.cfg["fusion_dim"]:
                if not hasattr(self, 'struct_to_fusion'):
                    self.struct_to_fusion = nn.Linear(self.cfg["kge_ent_dim"], self.cfg["fusion_dim"]).to(self.device)
                    nn.init.xavier_uniform_(self.struct_to_fusion.weight)
                    nn.init.zeros_(self.struct_to_fusion.bias)
                struct_emb = self.struct_to_fusion(struct_emb)
            
            # Flatten -> fuse -> reshape
            original_shape = struct_emb.shape[:-1]  # [B, ...]
            fusion_dim = struct_emb.shape[-1]
            
            struct_flat = struct_emb.view(-1, fusion_dim)
            visual_flat = visual_emb.view(-1, fusion_dim)
            textual_flat = textual_emb.view(-1, fusion_dim)
            numeric_flat = numeric_emb.view(-1, fusion_dim)
            rel_gate_flat = rel_gate.view(-1, 1)
            
            joint_flat = self._get_joint_embeddings(
                struct_emb=struct_flat,
                visual_emb=visual_flat,
                textual_emb=textual_flat,
                numeric_emb=numeric_flat,
                rel_gate=rel_gate_flat
            )  # [B*..., fusion_dim]
            
            joint_emb = joint_flat.view(*original_shape, fusion_dim)
            
            # Project to LLM space
            proj_emb = self.fusion_to_llm(joint_emb)  # [B, ..., llm_dim * num_prefix]
            proj_emb = proj_emb.view(*original_shape, self.cfg["num_prefix"], self.cfg["llm_dim"]).mean(dim=-2)  # [B, ..., llm_dim]
        else:
            # No relation-guided fusion: project struct embeddings only
            if struct_emb.size(-1) != self.cfg["fusion_dim"]:
                if not hasattr(self, 'struct_to_fusion'):
                    self.struct_to_fusion = nn.Linear(self.cfg["kge_ent_dim"], self.cfg["fusion_dim"]).to(self.device)
                    nn.init.xavier_uniform_(self.struct_to_fusion.weight)
                    nn.init.zeros_(self.struct_to_fusion.bias)
                struct_emb = self.struct_to_fusion(struct_emb)
            
            original_shape = struct_emb.shape[:-1]
            struct_flat = struct_emb.view(-1, self.cfg["fusion_dim"])
            proj_flat = self.fusion_to_llm(struct_flat)  # [B*..., llm_dim * num_prefix]
            proj_emb = proj_flat.view(*original_shape, self.cfg["num_prefix"], self.cfg["llm_dim"]).mean(dim=-2)  # [B, ..., llm_dim]
        
        proj_emb = self.prefix_norm(proj_emb)
        proj_emb = torch.clamp(proj_emb, -5.0, 5.0)

        # Default to float32 unless target_dtype is specified
        if target_dtype is not None:
            proj_emb = proj_emb.to(dtype=target_dtype)
        else:
            proj_emb = proj_emb.to(dtype=torch.float32)
        return proj_emb
