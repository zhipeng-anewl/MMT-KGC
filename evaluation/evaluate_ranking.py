import torch
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import os

import sys
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJ_ROOT)
from utils.config import MMTConfig


def _infer_entity_type(eid: int) -> int:
    """Infer entity type id from entity id."""
    if 0 <= eid <= 866:
        return 0
    elif 867 <= eid <= 1368:
        return 1
    elif 1369 <= eid <= 15097:
        return 2
    else:
        return 3


@torch.no_grad()
def _load_multimodal_kge_model(
    embeddings_dir: str = None,
    fusion_method: str = "add",
    device: torch.device = None
):
    """Load multimodal RotatE model for coarse ranking."""
    if embeddings_dir is None:
        embeddings_dir = f"{PROJ_ROOT}/data/multimodal_kge_models"
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        data_path = f"{PROJ_ROOT}/data/processed"
        with open(f'{data_path}/entity2id.txt', encoding='utf-8') as f:
            nentity = int(f.readline().strip())
        with open(f'{data_path}/relation2id.txt', encoding='utf-8') as f:
            nrelation = int(f.readline().strip())
        
        model_file = f"{embeddings_dir}/RotatE_best_model_64d_multimodal_{fusion_method}.pt"
        checkpoint = None
        use_full_model = False
        
        if os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
            use_full_model = True
            print(f"Loaded multimodal RotatE projection weights from .pt: {model_file}")
        else:
            print(f"RotatE checkpoint not found: {model_file}. Using randomly initialized projection layers.")
        
        if use_full_model and checkpoint:
            fusion_method = checkpoint.get('fusion_method', fusion_method)
            multimodal_weight = checkpoint.get('multimodal_weight', 0.5)
            model_gamma = checkpoint.get('gamma', 12.0)
            dim = checkpoint.get('dim', 64)
            double_entity_embedding = checkpoint.get('double_entity_embedding', True)
            double_relation_embedding = checkpoint.get('double_relation_embedding', False)
            print(f"Loaded config from checkpoint: fusion_method={fusion_method}, multimodal_weight={multimodal_weight}")
            print(f"double_entity_embedding={double_entity_embedding}, double_relation_embedding={double_relation_embedding}")
        else:
            multimodal_weight = 0.5
            model_gamma = 12.0
            dim = 64
            double_entity_embedding = True
            double_relation_embedding = False
            print(f"Checkpoint not found; using defaults: fusion_method={fusion_method}, multimodal_weight={multimodal_weight}")
        
        from models.multimodal_kge_model import MultimodalKGEModel
        
        multimodal_data_dir = MMTConfig.DATA_PATHS["embeddings"]
        kge_model = MultimodalKGEModel(
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
        
        entity_file = MMTConfig.FILE_PATHS["kge_entity"]
        relation_file = MMTConfig.FILE_PATHS["kge_relation"]
        
        print("Loading RotatE structural embeddings from config.py (kept consistent with the main model)")
        print(f"   entity embeddings: {entity_file}")
        print(f"   relation embeddings: {relation_file}")
        
        ent_embs = np.load(entity_file)
        rel_embs = np.load(relation_file)
        
        ent_embs = np.nan_to_num(ent_embs, nan=0.0, posinf=2.0, neginf=-2.0)
        rel_embs = np.nan_to_num(rel_embs, nan=0.0, posinf=2.0, neginf=-2.0)
        
        kge_model.entity_embedding.data = torch.from_numpy(ent_embs).float()
        kge_model.relation_embedding.data = torch.from_numpy(rel_embs).float()
        
        if use_full_model and checkpoint:
            proj_loaded = False
            if 'herb_image_proj' in checkpoint and hasattr(kge_model, 'herb_image_proj') and kge_model.herb_image_proj is not None:
                kge_model.herb_image_proj.load_state_dict(checkpoint['herb_image_proj'])
                proj_loaded = True
            if 'mol_graph_proj' in checkpoint and hasattr(kge_model, 'mol_graph_proj') and kge_model.mol_graph_proj is not None:
                kge_model.mol_graph_proj.load_state_dict(checkpoint['mol_graph_proj'])
                proj_loaded = True
            if 'mol_attr_proj' in checkpoint and hasattr(kge_model, 'mol_attr_proj') and kge_model.mol_attr_proj is not None:
                kge_model.mol_attr_proj.load_state_dict(checkpoint['mol_attr_proj'])
                proj_loaded = True
            
            if proj_loaded:
                print("Loaded multimodal projection layer weights (from RotatE checkpoint)")
            else:
                print("Projection layer weights not found; using randomly initialized projection layers")
        else:
            print("RotatE checkpoint not found; using randomly initialized projection layers")
        
        kge_model = kge_model.to(device)
        kge_model.eval()
        
        print(f"   nentity: {nentity}, nrelation: {nrelation}")
        print(f"   fusion_method: {fusion_method}, multimodal_weight: {multimodal_weight}")
        print(f"   device: {device}")
        
        return kge_model
    except Exception as e:
        print(f"Failed to load multimodal RotatE model: {e}")
        import traceback
        traceback.print_exc()
        print("   Falling back to full-entity ranking")
        raise


@torch.no_grad()
def _coarse_ranking_with_kge(
    h: int,
    r: int,
    all_entity_ids: List[int],
    kge_model,  # MultimodalKGEModel
    id2idx: Dict[int, int],
    gamma: float = 12.0,
    embedding_range: float = 2.0,
    filtered_tail_ids: Optional[set] = None
) -> Tuple[List[int], int]:
    """Coarse rank all entities with multimodal KGE; optionally apply filtered protocol."""
    _ = gamma, embedding_range  # kept for API compatibility (scores come from kge_model.forward)
    device = kge_model.entity_embedding.device
    
    head_part = torch.tensor([[h, r, 0]], device=device, dtype=torch.long)
    tail_part = torch.tensor([all_entity_ids], device=device, dtype=torch.long)
    
    sample = (head_part, tail_part)
    scores = kge_model(sample, mode='tail-batch')  # [1, N]
    scores = scores.squeeze(0)  # [N]
    
    filtered_count = 0
    if filtered_tail_ids is not None:
        for ft in filtered_tail_ids:
            if ft in id2idx:
                ft_idx = id2idx[ft]
                if ft_idx < len(scores):
                    scores[ft_idx] = -1e9
                    filtered_count += 1
    
    sorted_indices = torch.argsort(scores, descending=True)  # [N]
    full_ranking = [all_entity_ids[idx.item()] for idx in sorted_indices]
    
    return full_ranking, filtered_count


@torch.no_grad()
def evaluate_kg_ranking(
        model,
        dataset,
        device,
        max_eval_samples: int = None,
        random_seed: int = 42,
        all_true_triples: Optional[List[Tuple[int, int, int]]] = None,
        filtered: bool = True,
        evaluate_heads: bool = True,
        evaluate_tails: bool = True,
        two_stage: bool = False,
        rerank_top_m: int = 1000,
        kge_embeddings_dir: Optional[str] = None,
        kge_gamma: float = 12.0,
        kge_embedding_range: float = 2.0,
) -> Dict[str, float]:
    """
    KG ranking evaluation with optional two-stage mode:
      - coarse: multimodal RotatE ranks all entities
      - rerank: the LLM scores Top-m candidates and the final ranking is merged

    Metrics: MR / MRR / Hit@{1,3,10}.

    Args:
        model: MultiModalAdap model
        dataset: evaluation dataset
        device: torch device
        max_eval_samples: cap evaluation samples (None = full dataset)
        random_seed: RNG seed for sampling indices
        all_true_triples: required for filtered evaluation
        filtered: apply filtered protocol
        evaluate_heads / evaluate_tails: corruption side(s) to evaluate
        two_stage: enable coarse+rerank evaluation
        rerank_top_m: number of candidates to rerank with the LLM
    """
    import gc
    
    model.eval()
    dtype0 = next(model.parameters()).dtype
    model = model.to(torch.float32)
    all_ranks = []
    
    retriever_recall_stats = {
        "total_samples": 0,
        "correct_in_top_m": 0,
        "correct_not_in_top_m": 0,
    }
    rank_distribution = {
        "rank_le_1": 0,
        "rank_le_3": 0,
        "rank_le_10": 0,
        "rank_le_100": 0,
        "rank_gt_100": 0,
    }
    filtered_stats = {
        "total_filtered": 0,
        "filtered_in_candidates": 0,
        "filtered_in_coarse": 0,
    }

    all_entity_ids = list(dataset.id2name.keys())
    id2idx = {eid: idx for idx, eid in enumerate(all_entity_ids)}

    if all_true_triples is None:
        raise ValueError("all_true_triples is required for filtered evaluation")

    hr_to_tails = defaultdict(set)
    rt_to_heads = defaultdict(set)
    for h, r, t in all_true_triples:
        hr_to_tails[(h, r)].add(t)
        rt_to_heads[(r, t)].add(h)

    kge_model = None
    if two_stage:
        try:
            kge_model = _load_multimodal_kge_model(
                embeddings_dir=kge_embeddings_dir,
                fusion_method="add",
                device=device
            )
            print(f"Two-stage evaluation enabled: coarse (full-entity) + rerank (Top-{rerank_top_m})")
        except Exception as e:
            print(f"Failed to enable two-stage evaluation: {e}")
            print("   Falling back to full-entity ranking")
            two_stage = False

    indices = list(range(len(dataset)))
    if max_eval_samples is not None and max_eval_samples < len(indices):
        random.seed(random_seed)
        indices = random.sample(indices, max_eval_samples)
        indices.sort()
        print(f"Random sampling enabled (seed={random_seed}): sampled {max_eval_samples} / {len(dataset)}")
    else:
        print(f"Evaluating on all samples: {len(dataset)}")


    if two_stage:
        print(f"KG ranking eval (two-stage): samples={len(indices)}, rerank_top_m={rerank_top_m}")
    else:
        print(f"KG ranking eval (full-entity): samples={len(indices)}")

    if not (evaluate_heads or evaluate_tails):
        raise ValueError("At least one of evaluate_heads/evaluate_tails must be True")

    for idx in tqdm(indices, desc="KG Full-Entity Ranking (filtered)" if filtered else "KG Full-Entity Ranking"):
        sample = dataset[idx]

        # Ground-truth triple should come from dataset triples, because sample["embedding_ids"]
        # uses a tail placeholder (0) for prompt construction.
        if hasattr(dataset, "triples") and idx < len(dataset.triples):
            h, r, t = map(int, dataset.triples[idx])
        else:
            h, r, _ = sample["embedding_ids"].tolist()
            h, r = int(h), int(r)
            if "kg_candidate_ids" in sample and "kg_target_index" in sample:
                candidate_ids = sample["kg_candidate_ids"].view(-1)
                target_index = int(sample["kg_target_index"].view(-1)[0].item())
                if 0 <= target_index < candidate_ids.numel():
                    t = int(candidate_ids[target_index].item())
                else:
                    t = 0
            else:
                t = 0

        # Head type hint (prefer dataset-provided type if available)
        head_type_default = _infer_entity_type(h)
        if "entity_types" in sample:
            if isinstance(sample["entity_types"], (list, tuple)) and len(sample["entity_types"]) >= 1:
                head_type_default = sample["entity_types"][0]
            elif isinstance(sample["entity_types"], torch.Tensor):
                if sample["entity_types"].numel() > 0:
                    head_type_default = sample["entity_types"][0].item()

        # Cache text inputs for this sample
        base_input_ids = sample["input_ids"].unsqueeze(0).to(device)
        base_attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        base_labels = sample["labels"].unsqueeze(0).to(device)

        # ========== tail corruption： (h, r, t') ==========
        if evaluate_tails:
            filtered_tail_ids = set()
            if filtered:
                tails_true = hr_to_tails.get((h, r), set())
                filtered_tail_ids = set(tails_true)
                if t in filtered_tail_ids:
                    filtered_tail_ids.remove(t)
            if two_stage and kge_model is not None:
                retriever_full_ranking, coarse_filtered_count = _coarse_ranking_with_kge(
                    h=h,
                    r=r,
                    all_entity_ids=all_entity_ids,
                    kge_model=kge_model,
                    id2idx=id2idx,
                    gamma=kge_gamma,
                    embedding_range=kge_embedding_range,
                    filtered_tail_ids=filtered_tail_ids if filtered else None,
                )
                
                if filtered:
                    filtered_stats["filtered_in_coarse"] = filtered_stats.get("filtered_in_coarse", 0) + coarse_filtered_count
                
                top_m_candidates = retriever_full_ranking[:rerank_top_m]
                
                retriever_recall_stats["total_samples"] += 1
                
                if t not in retriever_full_ranking:
                    import warnings
                    warnings.warn(
                        f"Warning: true tail {t} not found in coarse full ranking. "
                        f"Sample: (h={h}, r={r}, t={t})"
                    )
                    retriever_recall_stats["correct_not_in_top_m"] += 1
                    top_m_candidates.append(t)
                elif t in top_m_candidates:
                    retriever_recall_stats["correct_in_top_m"] += 1
                else:
                    retriever_recall_stats["correct_not_in_top_m"] += 1
                    top_m_candidates.append(t)
                
                candidates = top_m_candidates
                num_candidates = len(candidates)
                candidate_to_idx = {cand: i for i, cand in enumerate(candidates)}

                h_tensor = torch.tensor([h], device=device)
                r_tensor = torch.tensor([r], device=device)
                t_tensor = torch.tensor([candidates[0]], device=device)  # placeholder
                embedding_ids = torch.stack([h_tensor, r_tensor, t_tensor], dim=1)  # [1, 3]
                
                entity_types = torch.tensor(
                    [[head_type_default, _infer_entity_type(candidates[0])]],
                    dtype=torch.long,
                    device=device,
                )  # [1, 2]
                
                kg_candidate_ids = torch.tensor([candidates], device=device)  # [1, N]
                
                outputs = model(
                    input_ids=base_input_ids,
                    attention_mask=base_attention_mask,
                    labels=base_labels,
                    embedding_ids=embedding_ids,
                    entity_types=entity_types,
                    kg_candidate_ids=kg_candidate_ids,
                    use_cache=False,
                    return_dict=True,
                )
                
                kg_scores_batch = outputs.get("kg_scores") if isinstance(outputs, dict) else getattr(outputs, "kg_scores", None)
                if kg_scores_batch is None:
                    scores_tensor = torch.randn(num_candidates, device=device)
                else:
                    if kg_scores_batch.dim() == 2 and kg_scores_batch.size(0) == 1:
                        scores_tensor = kg_scores_batch.squeeze(0)
                    elif kg_scores_batch.dim() == 1:
                        scores_tensor = kg_scores_batch
                    else:
                        scores_tensor = kg_scores_batch.flatten()[:num_candidates]
                
                scores = scores_tensor.detach().float().cpu().numpy().astype(np.float32)
                scores = np.clip(scores, -10.0, 10.0)
                
                if filtered and filtered_tail_ids:
                    filtered_stats["total_filtered"] += len(filtered_tail_ids)
                    for ft in filtered_tail_ids:
                        if ft in candidate_to_idx:
                            scores[candidate_to_idx[ft]] = -1e9
                            filtered_stats["filtered_in_candidates"] += 1
                
                sorted_indices_rerank = np.argsort(scores)[::-1]
                llm_reranked_top_m = [candidates[i] for i in sorted_indices_rerank[: min(rerank_top_m, len(candidates))]]
                llm_reranked_top_m_set = set(llm_reranked_top_m)
                
                retriever_remaining = []
                for eid in retriever_full_ranking[rerank_top_m:]:
                    if eid not in llm_reranked_top_m_set:
                        retriever_remaining.append(eid)
                
                final_ranking = llm_reranked_top_m + retriever_remaining
                pos_rank_tail = final_ranking.index(t) + 1 if t in final_ranking else (retriever_full_ranking.index(t) + 1)
            
            else:
                candidates = all_entity_ids
                num_candidates = len(candidates)
                candidate_to_idx = {cand: i for i, cand in enumerate(candidates)}

                h_tensor = torch.tensor([h], device=device)
                r_tensor = torch.tensor([r], device=device)
                t_tensor = torch.tensor([candidates[0]], device=device)  # placeholder
                embedding_ids = torch.stack([h_tensor, r_tensor, t_tensor], dim=1)  # [1, 3]

                entity_types = torch.tensor(
                    [[head_type_default, _infer_entity_type(candidates[0])]],
                    dtype=torch.long,
                    device=device,
                )  # [1, 2]

                kg_candidate_ids = torch.tensor([candidates], device=device)  # [1, N]

                outputs = model(
                    input_ids=base_input_ids,
                    attention_mask=base_attention_mask,
                    labels=base_labels,
                    embedding_ids=embedding_ids,
                    entity_types=entity_types,
                    kg_candidate_ids=kg_candidate_ids,
                    use_cache=False,
                    return_dict=True,
                )

                kg_scores_batch = outputs.get("kg_scores") if isinstance(outputs, dict) else getattr(outputs, "kg_scores", None)
                if kg_scores_batch is None:
                    scores_tensor = torch.randn(num_candidates, device=device)
                else:
                    if kg_scores_batch.dim() == 2 and kg_scores_batch.size(0) == 1:
                        scores_tensor = kg_scores_batch.squeeze(0)
                    elif kg_scores_batch.dim() == 1:
                        scores_tensor = kg_scores_batch
                    else:
                        scores_tensor = kg_scores_batch.flatten()[:num_candidates]

                scores = scores_tensor.detach().float().cpu().numpy().astype(np.float32)
                scores = np.clip(scores, -10.0, 10.0)

                if filtered and filtered_tail_ids:
                    filtered_stats["total_filtered"] += len(filtered_tail_ids)
                    for ft in filtered_tail_ids:
                        if ft in candidate_to_idx:
                            scores[candidate_to_idx[ft]] = -1e9
                            filtered_stats["filtered_in_candidates"] += 1

                sorted_indices = np.argsort(scores)[::-1]
                final_ranking = [candidates[i] for i in sorted_indices]
                pos_rank_tail = final_ranking.index(t) + 1 if t in final_ranking else len(final_ranking)
            
            if pos_rank_tail <= 1:
                rank_distribution["rank_le_1"] += 1
            if pos_rank_tail <= 3:
                rank_distribution["rank_le_3"] += 1
            if pos_rank_tail <= 10:
                rank_distribution["rank_le_10"] += 1
            if pos_rank_tail <= 100:
                rank_distribution["rank_le_100"] += 1
            else:
                rank_distribution["rank_gt_100"] += 1
        
            all_ranks.append(pos_rank_tail)
            
            del outputs, kg_scores_batch, scores_tensor, scores, embedding_ids, entity_types, kg_candidate_ids
            torch.cuda.empty_cache()

        if evaluate_heads:
            candidates_h = all_entity_ids
            num_candidates_h = len(candidates_h)

            filtered_head_ids = set()
            if filtered:
                heads_true = rt_to_heads.get((r, t), set())
                filtered_head_ids = set(heads_true)
                if h in filtered_head_ids:
                    filtered_head_ids.remove(h)
            h_tensor_h = torch.tensor(candidates_h, device=device)
            r_tensor_h = torch.tensor([r] * num_candidates_h, device=device)
            t_tensor_h = torch.tensor([t] * num_candidates_h, device=device)
            embedding_ids_h = torch.stack([h_tensor_h, r_tensor_h, t_tensor_h], dim=1)  # [B, 3]

            input_ids_h = base_input_ids.repeat(num_candidates_h, 1)
            attention_mask_h = base_attention_mask.repeat(num_candidates_h, 1)
            labels_h = base_labels.repeat(num_candidates_h, 1)
            
            tail_type_fixed = _infer_entity_type(t)
            entity_types_h = torch.tensor(
                [[_infer_entity_type(cand_h), tail_type_fixed] for cand_h in candidates_h],
                dtype=torch.long,
                device=device,
            )

            kg_candidate_ids_h = torch.tensor([[cand] for cand in candidates_h], device=device)  # [B, 1]
            
            outputs_h = model(
                input_ids=input_ids_h,
                attention_mask=attention_mask_h,
                labels=labels_h,
                embedding_ids=embedding_ids_h,
                entity_types=entity_types_h,
                kg_candidate_ids=kg_candidate_ids_h,
                use_cache=False,
                return_dict=True,
            )
            
            kg_scores_batch_h = outputs_h.get("kg_scores") if isinstance(outputs_h, dict) else getattr(outputs_h, "kg_scores", None)
            if kg_scores_batch_h is None:
                scores_h_tensor = torch.randn(num_candidates_h, device=device)
            else:
                if kg_scores_batch_h.dim() == 2 and kg_scores_batch_h.size(1) == 1:
                    scores_h_tensor = kg_scores_batch_h.squeeze(1)  # [B]
                elif kg_scores_batch_h.dim() == 2 and kg_scores_batch_h.size(0) == kg_scores_batch_h.size(1):
                    scores_h_tensor = kg_scores_batch_h.diagonal()
                else:
                    scores_h_tensor = kg_scores_batch_h.flatten()[:num_candidates_h]
            
            scores_h = scores_h_tensor.detach().float().cpu().numpy().astype(np.float32)
            scores_h = np.clip(scores_h, -10.0, 10.0)
                
            del outputs_h, kg_scores_batch_h, scores_h_tensor, input_ids_h, attention_mask_h, labels_h, embedding_ids_h, entity_types_h, kg_candidate_ids_h
            torch.cuda.empty_cache()

            if filtered and filtered_head_ids:
                for fh in filtered_head_ids:
                    if fh in id2idx:
                        scores_h[id2idx[fh]] = -1e9

            if h not in id2idx:
                pos_rank_head = float('inf')
            else:
                pos_idx_head = id2idx[h]
                sorted_indices_h = np.argsort(scores_h)[::-1]
                pos_rank_head = int(np.where(sorted_indices_h == pos_idx_head)[0][0]) + 1

            all_ranks.append(pos_rank_head)
            del scores_h

        if idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    if len(all_ranks) == 0:
        return {
            "mr": float("nan"),
            "mrr": float("nan"),
            "hit1": float("nan"),
            "hit3": float("nan"),
            "hit10": float("nan"),
        }

    ranks = np.array(all_ranks)

    metrics = {
        "mr": float(np.mean(ranks)),  # Mean Rank
        "mrr": float(np.mean(1.0 / ranks)),  # Mean Reciprocal Rank
        "hit1": float(np.mean(ranks <= 1)),  # Hit@1
        "hit3": float(np.mean(ranks <= 3)),  # Hit@3
        "hit10": float(np.mean(ranks <= 10)),  # Hit@10
    }

    print(f"\n{'='*70}")
    print(f"Evaluation results (samples: {len(ranks)})")
    print(f"{'='*70}")
    print(f"Mean Rank: {metrics['mr']:.2f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Hit@1: {metrics['hit1']:.4f}")
    print(f"Hit@3: {metrics['hit3']:.4f}")
    print(f"Hit@10: {metrics['hit10']:.4f}")
    print(f"Rank distribution: min={ranks.min()}, max={ranks.max()}, median={np.median(ranks)}")
    
    if two_stage and retriever_recall_stats["total_samples"] > 0:
        total_samples = retriever_recall_stats["total_samples"]
        correct_in_top_m = retriever_recall_stats["correct_in_top_m"]
        correct_not_in_top_m = retriever_recall_stats["correct_not_in_top_m"]
        
        if correct_in_top_m + correct_not_in_top_m != total_samples:
            import warnings
            warnings.warn(
                "Coarse recall stats mismatch: "
                f"correct_in_top_m ({correct_in_top_m}) + correct_not_in_top_m ({correct_not_in_top_m}) "
                f"!= total_samples ({total_samples})"
            )
        
        recall_at_m = correct_in_top_m / total_samples if total_samples > 0 else 0.0
        
        print(f"\n{'='*70}")
        print(f"Coarse recall stats (Top-{rerank_top_m})")
        print(f"{'='*70}")
        print(f"total_samples: {total_samples}")
        print(f"correct_in_top_{rerank_top_m}: {correct_in_top_m}")
        print(f"correct_not_in_top_{rerank_top_m}: {correct_not_in_top_m}")
        print(f"check: {correct_in_top_m} + {correct_not_in_top_m} = {correct_in_top_m + correct_not_in_top_m} (should equal {total_samples})")
        print(f"recall@Top-{rerank_top_m}: {recall_at_m:.4f} ({recall_at_m*100:.2f}%)")
        if recall_at_m > 0.9:
            print("High coarse recall: retriever is strong")
        elif recall_at_m > 0.7:
            print("Medium coarse recall: LLM reranking may help")
        else:
            print("Low coarse recall: may limit final performance")
    
    print(f"\n{'='*70}")
    print("Rank distribution (detailed)")
    print(f"{'='*70}")
    total_samples = len(ranks)
    print(f"rank ≤ 1: {rank_distribution['rank_le_1']} ({rank_distribution['rank_le_1']/total_samples*100:.2f}%)")
    print(f"rank ≤ 3: {rank_distribution['rank_le_3']} ({rank_distribution['rank_le_3']/total_samples*100:.2f}%)")
    print(f"rank ≤ 10: {rank_distribution['rank_le_10']} ({rank_distribution['rank_le_10']/total_samples*100:.2f}%)")
    print(f"rank ≤ 100: {rank_distribution['rank_le_100']} ({rank_distribution['rank_le_100']/total_samples*100:.2f}%)")
    print(f"rank > 100: {rank_distribution['rank_gt_100']} ({rank_distribution['rank_gt_100']/total_samples*100:.2f}%)")
    
    if filtered and filtered_stats["total_filtered"] > 0:
        print(f"\n{'='*70}")
        print("Filtered evaluation stats")
        print(f"{'='*70}")
        print(f"total_true_triples_to_filter: {filtered_stats['total_filtered']}")
        if two_stage:
            print(f"filtered_in_coarse: {filtered_stats.get('filtered_in_coarse', 0)}")
            coarse_filter_ratio = filtered_stats.get('filtered_in_coarse', 0) / filtered_stats["total_filtered"]
            print(f"coarse_filter_ratio: {coarse_filter_ratio:.4f} ({coarse_filter_ratio*100:.2f}%)")
        print(f"filtered_in_candidates (Top-{rerank_top_m}): {filtered_stats['filtered_in_candidates']}")
        if filtered_stats["filtered_in_candidates"] > 0:
            fine_filter_ratio = filtered_stats["filtered_in_candidates"] / filtered_stats["total_filtered"]
            print(f"candidate_filter_ratio: {fine_filter_ratio:.4f} ({fine_filter_ratio*100:.2f}%)")
        total_filtered = filtered_stats.get('filtered_in_coarse', 0) + filtered_stats['filtered_in_candidates']
        if total_filtered > 0:
            total_filter_ratio = total_filtered / filtered_stats["total_filtered"]
            print(f"total_filtered: {total_filtered} (coarse + candidates)")
            print(f"total_filter_ratio: {total_filter_ratio:.4f} ({total_filter_ratio*100:.2f}%)")
            print(f"Filtered evaluation applied: filtered {total_filtered} known-true triples")
        else:
            print("Note: no true triples were found in candidates to filter (may already be excluded by coarse ranking)")
    
    print(f"\n{'='*70}\n")
    
    model = model.to(dtype=dtype0)
    torch.cuda.empty_cache()
    gc.collect()

    if two_stage and retriever_recall_stats["total_samples"] > 0:
        recall_at_m = retriever_recall_stats["correct_in_top_m"] / retriever_recall_stats["total_samples"]
        metrics["retriever_recall_at_m"] = recall_at_m
    else:
        metrics["retriever_recall_at_m"] = None
    metrics["rank_distribution"] = rank_distribution
    metrics["filtered_stats"] = filtered_stats

    return metrics