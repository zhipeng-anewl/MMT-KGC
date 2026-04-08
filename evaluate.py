#!/usr/bin/env python3
"""
MMT-KGC final evaluation script.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Tuple

import torch
from transformers import AutoTokenizer

PROJ_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJ_ROOT)

from models.multimodal_adap import MultiModalAdap
from evaluation.evaluate_ranking import evaluate_kg_ranking
from utils.config import MMTConfig
from utils.memory_optimizer import MemoryOptimizer
from utils.prompter import Prompter
from data.data_loader import MMTMultiModalDataset


def parse_args():
    parser = argparse.ArgumentParser(description="MMT-KGC final evaluation")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join(PROJ_ROOT, "outputs", "multimodal_adap_7b", "checkpoint"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=MMTConfig.BASE_MODEL_PATH,
        help="Base LLM path",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=MMTConfig.DATA_PATHS["processed"],
        help="Data directory (contains train2id/valid2id/test2id)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJ_ROOT, "outputs", "multimodal_adap_7b", "evaluation_results"),
        help="Evaluation output directory",
    )
    parser.add_argument("--cuda", type=str, default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max evaluation samples (default: all)")
    parser.add_argument("--rerank_top_m", type=int, default=100, help="Top-m for LLM reranking")
    parser.add_argument("--max_length", type=int, default=128, help="Tokenizer max length")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kge_gamma", type=float, default=12.0, help="RotatE gamma")
    parser.add_argument("--kge_embedding_range", type=float, default=2.0, help="RotatE embedding_range")
    return parser.parse_args()


def _load_triples(path: str) -> List[Tuple[int, int, int]]:
    """
    Load triples and normalize to (h, r, t).

    Supported formats:
    - first line is sample count (single integer)
    - each triple line is h t r
    """
    triples: List[Tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        first_line = True
        for line in f:
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if first_line and len(parts) == 1:
                first_line = False
                continue
            if len(parts) >= 3:
                h, t, r = map(int, parts[:3])
                triples.append((h, r, t))
            first_line = False
    return triples


def _load_checkpoint_weights(model: MultiModalAdap, checkpoint_dir: str):
    checkpoint_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    adapter_weights = {}
    kg_head_weights = {}

    for key, value in checkpoint.items():
        if key.startswith("multimodal_adapter."):
            adapter_weights[key.replace("multimodal_adapter.", "")] = value
        elif key.startswith("kg_score_head."):
            kg_head_weights[key.replace("kg_score_head.", "")] = value

    if adapter_weights:
        model.multimodal_adapter.load_state_dict(adapter_weights, strict=False)
        print(f"Adapter weights loaded: {len(adapter_weights)} items")
    else:
        print("No multimodal_adapter.* weights found in checkpoint")

    if kg_head_weights:
        model.kg_score_head.load_state_dict(kg_head_weights, strict=False)
        print(f"KG score head weights loaded: {len(kg_head_weights)} items")
    else:
        print("No kg_score_head.* weights found in checkpoint")

    if not adapter_weights and not kg_head_weights:
        model.load_state_dict(checkpoint, strict=False)
        print("Fallback load used for checkpoint (strict=False)")


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    MMTConfig.setup_cuda()
    device = MMTConfig.get_device("inference")

    train_path = os.path.join(args.data_dir, "train2id.txt")
    valid_path = os.path.join(args.data_dir, "valid2id.txt")
    test_path = os.path.join(args.data_dir, "test2id.txt")
    for p in [args.checkpoint_dir, args.base_model, train_path, valid_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Path does not exist: {p}")

    print("=" * 70)
    print("MMT-KGC Final Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Base model: {args.base_model}")
    print(f"Data dir: {args.data_dir}")
    print(f"Device: {device}")

    print("\n[1/4] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_dir,
        use_fast=False,
        legacy=False,
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    tokenizer.padding_side = "left"

    memory_optimizer = MemoryOptimizer(strategy="lora")
    base_model, _ = memory_optimizer.get_optimized_model(args.base_model)

    print("[2/4] Building multimodal model and loading checkpoint weights...")
    multimodal_model = MultiModalAdap(
        model=base_model,
        num_prefix=1,
        kge_model_path=MMTConfig.DATA_PATHS["embeddings"],
        adapter_config={
            "visual_dim": 2048,
            "textual_dim": 768,
            "numeric_dim": 7,
            "kge_ent_dim": 128,
            "kge_rel_dim": 64,
            "fusion_dim": 128,
            "llm_dim": 4096,
            "num_prefix": 1,
        },
    )
    _load_checkpoint_weights(multimodal_model, args.checkpoint_dir)
    multimodal_model.to(device)
    multimodal_model.eval()

    print("[3/4] Building test dataset and filtered all-true triples...")
    prompter = Prompter("kg_completion")
    test_dataset = MMTMultiModalDataset(
        test_path,
        tokenizer,
        prompter=prompter,
        max_length=args.max_length,
        num_negatives=99,
        use_retriever_for_training=False,
    )

    train_triples = _load_triples(train_path)
    valid_triples = _load_triples(valid_path)
    test_triples = _load_triples(test_path)
    all_true_triples = train_triples + valid_triples + test_triples

    print(f"Test samples: {len(test_dataset)}")
    print(
        f"All true triples: train={len(train_triples)}, valid={len(valid_triples)}, test={len(test_triples)}"
    )

    print("[4/4] Running final two-stage evaluation...")
    metrics = evaluate_kg_ranking(
        model=multimodal_model,
        dataset=test_dataset,
        device=device,
        max_eval_samples=args.max_eval_samples,
        random_seed=args.random_seed,
        all_true_triples=all_true_triples,
        filtered=True,
        evaluate_heads=False,
        evaluate_tails=True,
        two_stage=True,
        rerank_top_m=args.rerank_top_m,
        kge_embeddings_dir=None,
        kge_gamma=args.kge_gamma,
        kge_embedding_range=args.kge_embedding_range,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "checkpoint": args.checkpoint_dir,
        "data_dir": args.data_dir,
        "test_path": test_path,
        "evaluation_config": {
            "max_eval_samples": args.max_eval_samples,
            "random_seed": args.random_seed,
            "filtered": True,
            "evaluate_heads": False,
            "evaluate_tails": True,
            "two_stage": True,
            "rerank_top_m": args.rerank_top_m,
            "kge_gamma": args.kge_gamma,
            "kge_embedding_range": args.kge_embedding_range,
            "max_length": args.max_length,
        },
        "metrics": metrics,
        "timestamp": timestamp,
    }

    result_file = os.path.join(
        args.output_dir, f"final_test_100k_checkpoint5000_{timestamp}.json"
    )
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True, indent=2)

    print("\n" + "=" * 70)
    print("Final evaluation completed")
    print("=" * 70)
    print(f"MR: {metrics['mr']:.2f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Hit@1: {metrics['hit1']:.4f}")
    print(f"Hit@3: {metrics['hit3']:.4f}")
    print(f"Hit@10: {metrics['hit10']:.4f}")
    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    main()
