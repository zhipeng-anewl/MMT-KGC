#!/usr/bin/env python3
"""Multimodal KGE model.

This wraps a base KGE model (e.g., RotatE) and fuses unified multimodal features
(visual/textual/numeric) into entity representations before scoring.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
import sys
import os

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJ_ROOT)

from models.kge_model import KGEModel
from utils.config import MMTConfig


class MultimodalKGEModel(KGEModel):
    """KGE model with multimodal feature fusion on entity embeddings."""
    
    def __init__(
        self,
        model_name: str,
        nentity: int,
        nrelation: int,
        hidden_dim: int,
        gamma: float,
        double_entity_embedding: bool = False,
        double_relation_embedding: bool = False,
        multimodal_data_dir: str = None,
        fusion_method: str = 'concat',  # 'concat' | 'add' | 'weighted'
        multimodal_weight: float = 0.5,
    ):
        """Create a multimodal-augmented KGE model."""
        super().__init__(
            model_name=model_name,
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=hidden_dim,
            gamma=gamma,
            double_entity_embedding=double_entity_embedding,
            double_relation_embedding=double_relation_embedding
        )
        
        self.fusion_method = fusion_method
        self.multimodal_weight = multimodal_weight
        
        if multimodal_data_dir is None:
            multimodal_data_dir = MMTConfig.DATA_PATHS["embeddings"]
        
        self._load_multimodal_data(multimodal_data_dir)
        
        self._init_multimodal_projectors()
        
        if fusion_method == 'concat':
            self._adjust_entity_embedding_for_concat()
    
    def _move_multimodal_data_to_device(self, device):
        """Kept for API compatibility (multimodal tensors are moved on demand)."""
        pass
    
    def _get_multimodal_embeddings_batch(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """Batch multimodal embeddings projected to entity_dim."""
        device = entity_ids.device
        safe_ids = entity_ids.clamp(0, self.visual_embs.size(0) - 1).cpu()

        visual_embs = self.visual_proj(self.visual_embs[safe_ids].to(device))
        textual_embs = self.textual_proj(self.textual_embs[safe_ids].to(device))
        numeric_embs = self.numeric_proj(self.numeric_embs[safe_ids].to(device))

        return visual_embs + textual_embs + numeric_embs
    
    def _load_multimodal_data(self, data_dir: str):
        """Load unified multimodal features: visual/textual/numeric."""
        visual_path = os.path.join(data_dir, "visual.pth")
        textual_path = os.path.join(data_dir, "textual.pth")
        numeric_path = os.path.join(data_dir, "numeric.pth")

        missing = [p for p in (visual_path, textual_path, numeric_path) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing multimodal feature files. Ensure data/embeddings contains visual.pth, textual.pth, numeric.pth."
            )

        visual = torch.load(visual_path)
        textual = torch.load(textual_path)
        numeric = torch.load(numeric_path)

        if not isinstance(visual, torch.Tensor):
            visual = torch.tensor(visual)
        if not isinstance(textual, torch.Tensor):
            textual = torch.tensor(textual)
        if not isinstance(numeric, torch.Tensor):
            numeric = torch.tensor(numeric)

        self.visual_embs = torch.nan_to_num(visual.float(), nan=0.0, posinf=2.0, neginf=-2.0)
        self.textual_embs = torch.nan_to_num(textual.float(), nan=0.0, posinf=2.0, neginf=-2.0)
        self.numeric_embs = torch.nan_to_num(numeric.float(), nan=0.0, posinf=2.0, neginf=-2.0)

        self.visual_dim = self.visual_embs.shape[1]
        self.textual_dim = self.textual_embs.shape[1]
        self.numeric_dim = self.numeric_embs.shape[1]

        print(
            f"Loaded multimodal features: visual={self.visual_embs.shape}, "
            f"textual={self.textual_embs.shape}, numeric={self.numeric_embs.shape}"
        )
    
    def _load_entity_type_mapping(self):
        """Load entity id -> type mapping (optional utility)."""
        entity2id_path = MMTConfig.FILE_PATHS["entity2id"]
        entity2type_path = MMTConfig.FILE_PATHS["entity2type"]
        self.entity_id_to_type = {}
        
        try:
            # entity name -> id
            entity_name_to_id = {}
            with open(entity2id_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entity_name = parts[0]
                        entity_id = int(parts[1])
                        entity_name_to_id[entity_name] = entity_id
            
            # entity name -> type
            with open(entity2type_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entity_name = parts[0]
                        entity_type = parts[1]
                        if entity_name in entity_name_to_id:
                            entity_id = entity_name_to_id[entity_name]
                            self.entity_id_to_type[entity_id] = entity_type
            
            print(f"Loaded entity type mapping: {len(self.entity_id_to_type)} entities")
        except Exception as e:
            print(f"Failed to load entity type mapping: {e}")
            self.entity_id_to_type = {}
    
    def _get_entity_type(self, entity_id: int) -> str:
        """Get entity type string if mapping is loaded."""
        return self.entity_id_to_type.get(entity_id, 'unknown')
    
    def _init_multimodal_projectors(self):
        """Initialize modality projectors to entity_dim."""
        entity_dim = self.entity_dim
        self.visual_proj = nn.Linear(self.visual_dim, entity_dim)
        self.textual_proj = nn.Linear(self.textual_dim, entity_dim)
        self.numeric_proj = nn.Linear(self.numeric_dim, entity_dim)

        # Compatibility with historical checkpoint field names
        self.herb_image_proj = self.visual_proj
        self.mol_graph_proj = self.textual_proj
        self.mol_attr_proj = self.numeric_proj
    
    def _adjust_entity_embedding_for_concat(self):
        """No-op: concat is handled dynamically in forward for compatibility."""
        pass
    
    def _get_multimodal_embedding(self, entity_id: int) -> Optional[torch.Tensor]:
        """Get a single entity's multimodal embedding (projected to entity_dim)."""
        device = self.entity_embedding.device

        if entity_id < 0:
            return torch.zeros(self.entity_dim, device=device)

        safe_id = min(entity_id, self.visual_embs.size(0) - 1)
        visual = self.visual_proj(self.visual_embs[safe_id].to(device))
        textual = self.textual_proj(self.textual_embs[safe_id].to(device))
        numeric = self.numeric_proj(self.numeric_embs[safe_id].to(device))
        return visual + textual + numeric
    
    def _fuse_embeddings(self, struct_emb: torch.Tensor, multimodal_emb: Optional[torch.Tensor]) -> torch.Tensor:
        """Fuse structural and multimodal embeddings."""
        if multimodal_emb is None:
            return struct_emb
        
        if self.fusion_method == 'concat':
            return torch.cat([struct_emb, multimodal_emb], dim=-1)
        elif self.fusion_method == 'add':
            return struct_emb + multimodal_emb
        elif self.fusion_method == 'weighted':
            return (1 - self.multimodal_weight) * struct_emb + self.multimodal_weight * multimodal_emb
        else:
            return struct_emb
    
    def forward(self, sample, mode='single'):
        """Forward that fuses multimodal features into entity embeddings before scoring."""
        if mode == 'single':
            batch_size = sample.size(0)
            
            head_ids = sample[:, 0]
            tail_ids = sample[:, 2]
            
            struct_head = torch.index_select(self.entity_embedding, dim=0, index=head_ids)
            struct_tail = torch.index_select(self.entity_embedding, dim=0, index=tail_ids)
            
            multimodal_heads = self._get_multimodal_embeddings_batch(head_ids)
            multimodal_tails = self._get_multimodal_embeddings_batch(tail_ids)
            
            if self.fusion_method == 'concat':
                head = torch.cat([struct_head.unsqueeze(1), multimodal_heads.unsqueeze(1)], dim=-1)
                tail = torch.cat([struct_tail.unsqueeze(1), multimodal_tails.unsqueeze(1)], dim=-1)
                head = head[:, :, :self.entity_dim]
                tail = tail[:, :, :self.entity_dim]
            elif self.fusion_method == 'add':
                head = (struct_head + multimodal_heads).unsqueeze(1)
                tail = (struct_tail + multimodal_tails).unsqueeze(1)
            elif self.fusion_method == 'weighted':
                head = ((1 - self.multimodal_weight) * struct_head + 
                       self.multimodal_weight * multimodal_heads).unsqueeze(1)
                tail = ((1 - self.multimodal_weight) * struct_tail + 
                       self.multimodal_weight * multimodal_tails).unsqueeze(1)
            else:
                head = struct_head.unsqueeze(1)
                tail = struct_tail.unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            struct_head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            struct_tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
            # Tail side (one per batch)
            tail_ids = tail_part[:, 2]  # [B]
            multimodal_tails = self._get_multimodal_embeddings_batch(tail_ids)  # [B, entity_dim]
            struct_tail_flat = struct_tail.squeeze(1)  # [B, entity_dim]
            
            if self.fusion_method == 'concat':
                tail = torch.cat([struct_tail_flat.unsqueeze(1), multimodal_tails.unsqueeze(1)], dim=-1)
                tail = tail[:, :, :self.entity_dim]
            elif self.fusion_method == 'add':
                tail = (struct_tail_flat + multimodal_tails).unsqueeze(1)
            elif self.fusion_method == 'weighted':
                tail = ((1 - self.multimodal_weight) * struct_tail_flat + 
                       self.multimodal_weight * multimodal_tails).unsqueeze(1)
            else:
                tail = struct_tail
            
            # Head side (many per batch)
            head_ids_flat = head_part.view(-1)  # [B * negative_sample_size]
            multimodal_heads_flat = self._get_multimodal_embeddings_batch(head_ids_flat)  # [B*N, entity_dim]
            multimodal_heads = multimodal_heads_flat.view(batch_size, negative_sample_size, -1)  # [B, N, entity_dim]
            struct_head_flat = struct_head.view(batch_size, negative_sample_size, -1)  # [B, N, entity_dim]
            
            if self.fusion_method == 'concat':
                head = torch.cat([struct_head_flat, multimodal_heads], dim=-1)
                head = head[:, :, :self.entity_dim]
            elif self.fusion_method == 'add':
                head = struct_head_flat + multimodal_heads
            elif self.fusion_method == 'weighted':
                head = (1 - self.multimodal_weight) * struct_head_flat + self.multimodal_weight * multimodal_heads
            else:
                head = struct_head_flat
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            struct_head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            struct_tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            # Head side (one per batch)
            head_ids = head_part[:, 0]  # [B]
            multimodal_heads = self._get_multimodal_embeddings_batch(head_ids)  # [B, entity_dim]
            struct_head_flat = struct_head.squeeze(1)  # [B, entity_dim]
            
            if self.fusion_method == 'concat':
                head = torch.cat([struct_head_flat.unsqueeze(1), multimodal_heads.unsqueeze(1)], dim=-1)
                head = head[:, :, :self.entity_dim]
            elif self.fusion_method == 'add':
                head = (struct_head_flat + multimodal_heads).unsqueeze(1)
            elif self.fusion_method == 'weighted':
                head = ((1 - self.multimodal_weight) * struct_head_flat + 
                       self.multimodal_weight * multimodal_heads).unsqueeze(1)
            else:
                head = struct_head
            
            # Tail side (many per batch)
            tail_ids_flat = tail_part.view(-1)  # [B * negative_sample_size]
            multimodal_tails_flat = self._get_multimodal_embeddings_batch(tail_ids_flat)  # [B*N, entity_dim]
            multimodal_tails = multimodal_tails_flat.view(batch_size, negative_sample_size, -1)  # [B, N, entity_dim]
            struct_tail_flat = struct_tail.view(batch_size, negative_sample_size, -1)  # [B, N, entity_dim]
            
            if self.fusion_method == 'concat':
                tail = torch.cat([struct_tail_flat, multimodal_tails], dim=-1)
                tail = tail[:, :, :self.entity_dim]
            elif self.fusion_method == 'add':
                tail = struct_tail_flat + multimodal_tails
            elif self.fusion_method == 'weighted':
                tail = (1 - self.multimodal_weight) * struct_tail_flat + self.multimodal_weight * multimodal_tails
            else:
                tail = struct_tail_flat
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
        
        else:
            raise ValueError('mode %s not supported' % mode)
        
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

