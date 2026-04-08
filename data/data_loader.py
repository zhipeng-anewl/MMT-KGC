# data/data_loader.py
import torch
import numpy as np
from torch.utils.data import Dataset,default_collate
from typing import Dict, Any, List, Optional
from utils.config import MMTConfig
from torch.nn.utils.rnn import pad_sequence
import os

class MMTMultiModalDataset(Dataset):
    """Link prediction dataset with optional multimodal KGE retriever."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        prompter=None,
        max_length: int = 256,
        num_negatives: int = 63,
        use_retriever_for_training: bool = True,
        kge_embeddings_dir: Optional[str] = None,
        kge_gamma: float = 12.0,
        kge_embedding_range: float = 2.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            data_path: path to train2id / valid2id / test2id.txt
            tokenizer: tokenizer of base LLM
            prompter: optional prompt template helper
            num_negatives: number of negative tails per positive
            use_retriever_for_training: whether to use multimodal KGE retriever
            kge_embeddings_dir: directory of multimodal KGE checkpoints
            kge_gamma / kge_embedding_range: RotatE hyper-parameters
            device: device for KGE model (default CPU)
        """

        self.tokenizer = tokenizer
        self.prompter = prompter
        self.max_length = max_length
        self.num_negatives = num_negatives
        self.use_retriever_for_training = use_retriever_for_training
        self.kge_gamma = kge_gamma
        self.kge_embedding_range = kge_embedding_range

        id2name_file = MMTConfig.FILE_PATHS["entityid2name"]
        self.id2name: Dict[int, str] = {}
        with open(id2name_file, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                idx, name = line.strip().split("\t", 1)
                self.id2name[int(idx)] = name

        self.all_entity_ids = np.array(list(self.id2name.keys()))
        
        self.id2idx: Dict[int, int] = {eid: i for i, eid in enumerate(self.all_entity_ids)}

        self.triples = self._load_triples(data_path)
        self.triple_frequencies = self._count_triple_frequencies(self.triples)

        self.kge_model = None
        if self.use_retriever_for_training:
            self._load_kge_model(kge_embeddings_dir, device)


    # ----------------------------------------------------
    def _load_triples(self, path: str):
        triples = []
        with open(path, encoding='utf-8') as f:
            first_line = True
            for line in f:
                text = line.strip()
                if not text:
                    continue
                parts = text.split()
                if first_line and len(parts) == 1:
                    first_line = False
                    continue
                h, t, r = map(int, parts[:3])
                triples.append((h, r, t))
                first_line = False
        return triples

    def __len__(self):
        return len(self.triples)
    
    # ----------------------------------------------------
    def _count_triple_frequencies(self, triples: List[tuple]) -> Dict[tuple, int]:
        """Compute per-triple frequency for subsampling weights."""
        count = {}
        start = 4  # match baseline; avoid divide-by-zero
        
        for head, relation, tail in triples:
            key1 = (head, relation)
            if key1 not in count:
                count[key1] = start
            else:
                count[key1] += 1
            
            key2 = (tail, -relation - 1)
            if key2 not in count:
                count[key2] = start
            else:
                count[key2] += 1
        
        return count
    
    # ----------------------------------------------------
    def _load_kge_model(self, embeddings_dir: Optional[str] = None, device: Optional[torch.device] = None):
        """Load multimodal RotatE model for candidate retrieval."""
        if embeddings_dir is None:
            embeddings_dir = os.path.join(MMTConfig.BASE_PATH, "data", "multimodal_kge_models")
        
        # Multimodal feature directory from MMTConfig
        multimodal_data_dir = MMTConfig.DATA_PATHS["embeddings"]
        
        if device is None:
            device = torch.device("cpu")
        
        try:
            data_path = MMTConfig.DATA_PATHS["processed"]
            with open(f'{data_path}/entity2id.txt', encoding='utf-8') as f:
                nentity = int(f.readline().strip())
            with open(f'{data_path}/relation2id.txt', encoding='utf-8') as f:
                nrelation = int(f.readline().strip())
            
            from models.multimodal_kge_model import MultimodalKGEModel
            
            # Try loading RotatE checkpoint (projection layers + config)
            model_file = f"{embeddings_dir}/RotatE_best_model_64d_multimodal_add.pt"
            checkpoint = None
            use_full_model = False
            
            if os.path.exists(model_file):
                checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                use_full_model = True
                print(f"Loaded multimodal RotatE projection weights from .pt: {model_file}")
            else:
                print(f"RotatE checkpoint not found: {model_file}. Using randomly initialized projection layers.")
            
            if use_full_model and checkpoint:
                fusion_method = checkpoint.get('fusion_method', 'add')
                multimodal_weight = checkpoint.get('multimodal_weight', 0.5)
                model_gamma = checkpoint.get('gamma', 12.0)
                dim = checkpoint.get('dim', 64)
                double_entity_embedding = checkpoint.get('double_entity_embedding', True)
                double_relation_embedding = checkpoint.get('double_relation_embedding', False)
                print(f"Loaded config from checkpoint: fusion_method={fusion_method}, multimodal_weight={multimodal_weight}")
                print(f"double_entity_embedding={double_entity_embedding}, double_relation_embedding={double_relation_embedding}")
            else:
                fusion_method = 'add'
                multimodal_weight = 0.5
                model_gamma = 12.0
                dim = 64
                double_entity_embedding = True
                double_relation_embedding = False
                print(f"Checkpoint not found; using defaults: fusion_method={fusion_method}, multimodal_weight={multimodal_weight}")
            
            self.kge_model = MultimodalKGEModel(
                model_name='RotatE',
                nentity=nentity,
                nrelation=nrelation,
                hidden_dim=dim,
                gamma=model_gamma,
                double_entity_embedding=double_entity_embedding,
                double_relation_embedding=double_relation_embedding,
                multimodal_data_dir=multimodal_data_dir,
                fusion_method=fusion_method,
                multimodal_weight=multimodal_weight
            )
            
            with torch.no_grad():
                entity_file = MMTConfig.FILE_PATHS["kge_entity"]
                relation_file = MMTConfig.FILE_PATHS["kge_relation"]
                
                print("Loading RotatE structural embeddings from config.py (kept consistent with the main model)")
                print(f"   entity embeddings: {entity_file}")
                print(f"   relation embeddings: {relation_file}")

                if not os.path.exists(entity_file) or not os.path.exists(relation_file):
                    raise FileNotFoundError(
                        "Missing structural embedding files. Please run training/train_kge.py to generate "
                        "RotatE_entity_64d_relation_aware_secondtimes.npy and "
                        "RotatE_relation_64d_relation_aware_secondtimes.npy."
                    )
                
                ent_embs = np.load(entity_file)
                rel_embs = np.load(relation_file)
                
                ent_embs = np.nan_to_num(ent_embs, nan=0.0, posinf=2.0, neginf=-2.0)
                rel_embs = np.nan_to_num(rel_embs, nan=0.0, posinf=2.0, neginf=-2.0)
                
                self.kge_model.entity_embedding.data = torch.from_numpy(ent_embs).float()
                self.kge_model.relation_embedding.data = torch.from_numpy(rel_embs).float()
                
                if use_full_model and checkpoint:
                    proj_loaded = False
                    if 'herb_image_proj' in checkpoint and hasattr(self.kge_model, 'herb_image_proj') and self.kge_model.herb_image_proj is not None:
                        self.kge_model.herb_image_proj.load_state_dict(checkpoint['herb_image_proj'])
                        proj_loaded = True
                    if 'mol_graph_proj' in checkpoint and hasattr(self.kge_model, 'mol_graph_proj') and self.kge_model.mol_graph_proj is not None:
                        self.kge_model.mol_graph_proj.load_state_dict(checkpoint['mol_graph_proj'])
                        proj_loaded = True
                    if 'mol_attr_proj' in checkpoint and hasattr(self.kge_model, 'mol_attr_proj') and self.kge_model.mol_attr_proj is not None:
                        self.kge_model.mol_attr_proj.load_state_dict(checkpoint['mol_attr_proj'])
                        proj_loaded = True
                    
                    if proj_loaded:
                        print("Loaded multimodal projection layer weights (from RotatE checkpoint)")
                    else:
                        print("Projection layer weights not found; using randomly initialized projection layers")
                else:
                    print("RotatE checkpoint not found; using randomly initialized projection layers")
            
            self.kge_model = self.kge_model.to(device)
            self.kge_model.eval()
            
            print(f"   nentity: {nentity}, nrelation: {nrelation}")
            print(f"   fusion_method: {fusion_method}, multimodal_weight: {multimodal_weight}")
            print(f"   device: {device}")
            
        except Exception as e:
            print(f"Failed to load multimodal RotatE model: {e}")
            import traceback
            traceback.print_exc()
            print("   Falling back to random negative sampling")
            self.use_retriever_for_training = False
            self.kge_model = None
    
    # ----------------------------------------------------
    @torch.no_grad()
    def _retrieve_candidates_with_kge(self, h: int, r: int, t_pos: int) -> List[int]:
        """Listwise candidate sampling using pretrained multimodal RotatE."""
        if self.kge_model is None:
            return self._random_sample_negatives(t_pos)
        
        device = next(self.kge_model.parameters()).device
        
        head_part = torch.tensor([[h, r, 0]], dtype=torch.long).to(device)  # [1, 3]
        tail_part = torch.tensor([self.all_entity_ids.tolist()], dtype=torch.long).to(device)  # [1, N]
        
        sample = (head_part, tail_part)
        scores = self.kge_model(sample, mode='tail-batch')  # [1, N]
        scores = scores.squeeze(0)  # [N]
        
        candidate_count = 1 + self.num_negatives
        top_m_size = candidate_count
        t_pos_idx = self.id2idx.get(t_pos, -1)
        if t_pos_idx >= 0 and t_pos_idx < len(scores):
            scores[t_pos_idx] = -1e9  # exclude positive tail
        
        sorted_indices = torch.argsort(scores, descending=True)  # [N]
        
        top_m_candidates = []
        for idx in sorted_indices:
            eid = self.all_entity_ids[idx.item()]
            if eid != t_pos:  # exclude positive tail
                top_m_candidates.append(int(eid))
                if len(top_m_candidates) >= top_m_size:  # dynamic Top-M
                    break
        
        import random
        sampled_negatives = random.sample(top_m_candidates, min(self.num_negatives, len(top_m_candidates)))
        
        candidates = [t_pos] + sampled_negatives
        random.shuffle(candidates)
        
        return candidates
    
    # ----------------------------------------------------
    def _random_sample_negatives(self, t_pos: int) -> List[int]:
        """Fallback: random negative sampling over all entities."""
        neg_tails = []
        while len(neg_tails) < self.num_negatives:
            neg = int(np.random.choice(self.all_entity_ids))
            if neg != t_pos:
                neg_tails.append(neg)
        
        candidates = [t_pos] + neg_tails
        import random
        random.shuffle(candidates)
        
        return candidates

    # ----------------------------------------------------
    def _infer_entity_type(self, eid: int) -> int:
        if 0 <= eid <= 866:
            return 0
        elif 867 <= eid <= 1368:
            return 1
        elif 1369 <= eid <= 15097:
            return 2
        else:
            return 3

    # ----------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        h, r, t_pos = self.triples[idx]

        h_name = self.id2name[h]
        r_name = MMTConfig.RELATION_TYPES.get(r, str(r))


        t_pos_name = self.id2name[t_pos]

        candidate_entity_ids = self._retrieve_candidates_with_kge(h, r, t_pos) if self.use_retriever_for_training else self._random_sample_negatives(t_pos)
        target_index = candidate_entity_ids.index(t_pos)  # 0-based
        
        candidate_names = [self.id2name[eid] for eid in candidate_entity_ids]
        
        if self.prompter is not None:
            listwise_prompt = self.prompter.generate_prompt(
                head_entity=h_name,
                relation=r_name,
                candidate_entities=candidate_names
            )
        else:
            prompt_parts = [
                "Given a head entity and a relation, select the correct tail entity",
                "from the candidate list.",
                "",
                "Head Entity:",
                h_name,
                "",
                "Relation:",
                r_name,
                "",
                "Candidates:"
            ]
            for i, cand_name in enumerate(candidate_names, start=1):
                prompt_parts.append(f"({i}) {cand_name}")
            prompt_parts.extend(["", "Answer with the index only:"])
            listwise_prompt = "\n".join(prompt_parts)
        
        target_label = target_index + 1  # convert to 1-based for the prompt

        enc = self.tokenizer(
            listwise_prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        labels = [-100] * len(enc["input_ids"])
        
        answer_text = str(target_label)
        answer_tokens = self.tokenizer.encode(answer_text, add_special_tokens=False)
        
        for token_id in answer_tokens:
            enc["input_ids"].append(token_id)
            enc["attention_mask"].append(1)
            labels.append(token_id)  # compute loss on the answer tokens
        
        if enc["input_ids"][-1] != self.tokenizer.eos_token_id:
            enc["input_ids"].append(self.tokenizer.eos_token_id)
            enc["attention_mask"].append(1)
            labels.append(self.tokenizer.eos_token_id)  # compute loss on EOS as well

        frequency = (
            self.triple_frequencies.get((h, r), 4) + 
            self.triple_frequencies.get((t_pos, -r - 1), 4)
        )
        subsampling_weight = np.sqrt(1.0 / frequency)  # match baseline

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),

            "kg_candidate_ids": torch.tensor(candidate_entity_ids, dtype=torch.long).unsqueeze(0),
            "kg_target_index": torch.tensor(target_index, dtype=torch.long).unsqueeze(0),

            "embedding_ids": torch.tensor([h, r, 0], dtype=torch.long),  # use 0 as tail placeholder
            "entity_types": [self._infer_entity_type(h), 0],  # tail_type placeholder
            
            "subsampling_weight": torch.tensor([subsampling_weight], dtype=torch.float32),
        }


# ----------------------------------------------------
def collate_fn(batch):
    """Pad 1-D sequences and stack tensors; return single item for batch_size=1."""
    if len(batch) == 1:
        return batch[0]

    keys = batch[0].keys()
    padded = {}
    for k in keys:
        vals = [b[k] for b in batch]
        # 1) pad 1-D tensors
        if isinstance(vals[0], torch.Tensor) and vals[0].ndim == 1:
            padded[k] = pad_sequence(vals, batch_first=True, padding_value=0)
        # 2) stack other tensors
        elif isinstance(vals[0], torch.Tensor):
            padded[k] = torch.stack(vals, dim=0)
        # 3) keep non-tensors as lists (HF Trainer can handle them)
        else:
            padded[k] = vals
    return padded