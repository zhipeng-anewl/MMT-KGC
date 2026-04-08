"""
MMT-KGC main training entry script.
Defaults to data/processed and llm/llama-2-7b.
"""

import os
import sys
import argparse

PROJ_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJ_ROOT)

from training.trainer import train_multimodal_adap
from utils.config import MMTConfig
from utils.memory_monitor import MemoryMonitor
import torch


def setup_environment(cuda_devices: str = "0"):
    """Set runtime environment.
    
    Args:
        cuda_devices: value for CUDA_VISIBLE_DEVICES, e.g. "0", "1", "0,1"
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        print(f"CUDA available. Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")


def parse_args():
    parser = argparse.ArgumentParser(description="MMT-KGC main training")

    parser.add_argument("--model_size", type=str, default="7b",
                        choices=["1b", "3b", "7b", "13b"],
                        help="Model size (default: 7b)")

    parser.add_argument("--strategy", type=str, default="lora",
                        choices=["lora"],
                        help="Memory optimization strategy (default: lora)")

    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size override")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Max sequence length override")
    parser.add_argument("--data_path", type=str, default=MMTConfig.DATA_PATHS["processed"],
                        help="Data directory (default: data/processed)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Optional base model path override")

    parser.add_argument("--cuda", type=str, default="0",
                        help="CUDA_VISIBLE_DEVICES value, e.g. '0', '1', '0,1'")

    return parser.parse_args()


def get_model_path(model_size: str) -> str:
    """Resolve model path by model size."""
    base_path = os.path.join(PROJ_ROOT, "llm")

    model_paths = {
        "1b": os.path.join(base_path, "llama-3-2-1b"),
        "3b": os.path.join(base_path, "llama-3-2-3b"),
        "7b": os.path.join(base_path, "llama-2-7b"),
        "13b": os.path.join(base_path, "llama-2-13b"),
    }

    path = model_paths.get(model_size)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path does not exist: {path}")

    return path


def adjust_training_config(model_size: str, strategy: str, args):
    """Build training config from model size and CLI options."""
    model_dimensions = {
        "1b": 2048,
        "3b": 3072,
        "7b": 4096,
        "13b": 5120,
    }
    llm_dim = model_dimensions.get(model_size, 4096)
    
    data_path = args.data_path
    base_model_path = args.base_model or get_model_path(model_size)
    
    config = {
        "base_model": base_model_path,
        "data_path": data_path,
        "output_dir": os.path.join(MMTConfig.OUTPUT_BASE_PATH, f"multimodal_adap_{model_size}"),

        "save_steps": 1000,
        "eval_steps": 1000,
        "per_device_eval_batch_size": 1,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_epochs": 3,
        "learning_rate": 1e-5,
        "cutoff_len": 128,
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        
        "lr_scheduler_type": "constant",
        
        # LoRA config
        "lora_r": 16 if model_size in ["1b", "3b"] else 8,
        "lora_alpha": 32 if model_size in ["1b", "3b"] else 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        
        "fp16": False,
        "logging_steps": 10,
        "save_total_limit": 3,
        "dataloader_num_workers": 0,
        
        # KG settings
        "kg_margin": 1.0,
        "kg_margin_lambda": 0.5,
        "listwise_weight": 1.0,
        "top_k_loss_weight": 0.0,
        "lm_loss_weight": 0.0,
        
        # Multimodal KGE retriever settings
        "kge_model_path": MMTConfig.DATA_PATHS["embeddings"],
        "num_negatives": 99,
        "kge_embeddings_dir": os.path.join(MMTConfig.BASE_PATH, "data", "multimodal_kge_models"),
        "kge_gamma": 12.0,
        "kge_embedding_range": 2.0,
        
        # Adapter config
        "adapter_config": {
            "visual_dim": 2048,
            "textual_dim": 768,
            "numeric_dim": 7,
            "kge_ent_dim": 128,
            "kge_rel_dim": 64,
            "fusion_dim": 128,
            "llm_dim": llm_dim,
            "num_prefix": 1,
        },
        
        "prompt_template": "kg_completion",
    }
    
    if args.batch_size is not None:
        config["micro_batch_size"] = args.batch_size
    if args.max_length is not None:
        config["cutoff_len"] = args.max_length
    
    if model_size in ["1b", "3b"]:
        config["gradient_accumulation_steps"] = 8
    elif model_size == "7b":
        config["gradient_accumulation_steps"] = 8
    else:
        config["gradient_accumulation_steps"] = 16

    return config


def main():
    args = parse_args()

    setup_environment(cuda_devices=args.cuda)

    monitor = MemoryMonitor(interval=10.0)
    monitor.start()

    try:
        training_config = adjust_training_config(
            model_size=args.model_size,
            strategy=args.strategy,
            args=args
        )

        print(f"\n{'='*70}")
        print("Start MMT-KGC training")
        print(f"{'='*70}")
        print(f"Model size: LLaMA-{args.model_size.upper()}")
        print(f"Strategy: {args.strategy}")
        print(f"Data path: {training_config['data_path']}")
        print(f"Base model: {training_config['base_model']}")
        print(f"Batch size: {training_config['micro_batch_size']}")
        print(f"Max length: {training_config['cutoff_len']}")
        print(f"LoRA rank: {training_config['lora_r']}")
        print(f"Output dir: {training_config['output_dir']}")
        print(f"Note: {MMTConfig.KGE_TRAINING_NOTE}")
        print(f"{'='*70}\n")

        os.makedirs(training_config['output_dir'], exist_ok=True)

        model = train_multimodal_adap(
            base_model=training_config["base_model"],
            data_path=training_config["data_path"],
            output_dir=training_config["output_dir"],
            training_config=training_config
        )

        print("\nTraining completed.")

    except torch.cuda.OutOfMemoryError:
        print("\nCUDA out of memory.")
        print("Suggestions:")
        print("1. Reduce batch size: --batch_size 1")
        print("2. Reduce sequence length: --max_length 64")
        print("3. Use a smaller model: --model_size 1b")
        print("4. Increase gradient accumulation steps in training config")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()

