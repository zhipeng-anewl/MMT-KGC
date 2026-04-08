# training/trainer.py
"""
Trainer for MultiModalAdap.

Implements KG ranking loss and ranking evaluation (optionally two-stage).
"""
import os
import torch
import gc
import random
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from typing import Dict, Any, Optional
from models.multimodal_adap import MultiModalAdap
from utils.config import MMTConfig
import logging
from data.data_loader import MMTMultiModalDataset, collate_fn
from utils.memory_optimizer import MemoryOptimizer
from evaluation.evaluate_ranking import evaluate_kg_ranking
from torch.utils.data import Subset, DataLoader

logger = logging.getLogger("trainer")


class LearningRateStepSchedulerCallback(TrainerCallback):
    """Step-based LR switch for constant scheduler setups."""
    
    def __init__(self, lr_step: int = 1000, new_lr: float = 1e-5):
        """
        Args:
            lr_step: global step at which to switch LR
            new_lr: LR after the switch
        """
        self.lr_step = lr_step
        self.new_lr = new_lr
        self.lr_changed = False
        self.trainer_ref = None
        
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Cache trainer reference if provided."""
        self.trainer_ref = kwargs.get('trainer')
        return control
        
    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        """
        Switch LR once at lr_step when using a constant scheduler.
        """
        if state.global_step == self.lr_step and not self.lr_changed:
            opt = None
            if optimizer is not None:
                opt = optimizer
            elif self.trainer_ref is not None and hasattr(self.trainer_ref, 'optimizer'):
                opt = self.trainer_ref.optimizer
            else:
                opt = kwargs.get('optimizer')
            
            if opt is not None:
                for param_group in opt.param_groups:
                    old_lr = param_group['lr']
                    param_group['lr'] = self.new_lr
                    logger.info(
                        f"LR update at step {state.global_step}: {old_lr:.2e} -> {self.new_lr:.2e}"
                    )
                self.lr_changed = True
            else:
                logger.warning(f"Step {state.global_step}: failed to locate optimizer; LR not updated")
        return control

class MultiModalTrainer(Trainer):
    """Trainer wrapper with project-specific evaluation and logging."""

    def __init__(self, training_config: Optional[Dict[str, Any]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = MMTConfig.get_device("train")
        self.logger = MMTConfig.get_trainer_logger()
        self.eval_logger = MMTConfig.get_eval_logger()
        self.training_config = training_config or {}


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Main objective: KG ranking loss.
        Optionally add a (small) LM loss term for regularization:
            L = w_kg * L_KG + w_lm * L_LM
        Defaults keep KG loss dominant.
        """

        device = self.device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        if hasattr(model, "llama_model") and hasattr(model.llama_model, "config"):
            model.llama_model.config.use_cache = False

        outputs = None
        loss = None
        kg_loss = None
        lm_loss = None

        try:
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
                embedding_ids=inputs.get("embedding_ids"),
                entity_types=inputs.get("entity_types"),
                kg_candidate_ids=inputs.get("kg_candidate_ids"),
                use_cache=False,
                return_dict=True,
            )
            if isinstance(outputs, dict):
                kg_scores = outputs.get("kg_scores", None)
                lm_loss = outputs.get("loss", None)
            else:
                kg_scores = getattr(outputs, "kg_scores", None)
                lm_loss = getattr(outputs, "loss", None)

            # =====================================================
            # KG ranking loss
            # =====================================================
            if kg_scores is not None:
                scores = kg_scores  # [B, N] where N = 1 + num_negatives

                kg_target_index = inputs.get("kg_target_index", None)
                if kg_target_index is not None:
                    if scores.dim() != 2:
                        logger.warning(
                            f"Unexpected scores shape: {scores.shape}, expected [B, N]. "
                            f"kg_target_index shape: {kg_target_index.shape}"
                        )
                    if scores.size(1) < 2:
                        logger.warning(
                            f"Too few candidates: scores.shape={scores.shape}, "
                            f"kg_target_index={kg_target_index}, "
                            f"kg_candidate_ids shape: {inputs.get('kg_candidate_ids', 'N/A')}"
                        )

                if torch.isnan(scores).any() or torch.isinf(scores).any():
                    logger.warning(
                        f"kg_scores contain NaN/Inf before loss calculation. "
                        f"NaN count: {torch.isnan(scores).sum().item()}, "
                        f"Inf count: {torch.isinf(scores).sum().item()}"
                    )
                    scores = torch.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=-10.0)

                scores = torch.clamp(scores, min=-10.0, max=10.0)

                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)  # [N] -> [1, N]

                # Listwise softmax loss (+ optional margin/top-k terms).
                if scores.size(1) < 2:
                    logger.warning(f"Not enough candidates (got {scores.size(1)}, need at least 2), using fallback loss")
                    loss = torch.tensor(0.1, device=device, requires_grad=True)
                else:
                    kg_target_index = inputs.get("kg_target_index", None)
                    
                    if kg_target_index is not None:
                        batch_size = scores.size(0)
                        num_candidates = scores.size(1)
                        
                        if kg_target_index.dim() == 2:
                            kg_target_index = kg_target_index.squeeze(1)
                        elif kg_target_index.dim() == 1:
                            if kg_target_index.size(0) == 1 and batch_size > 1:
                                kg_target_index = kg_target_index.expand(batch_size)
                        elif kg_target_index.dim() == 0:
                            kg_target_index = kg_target_index.unsqueeze(0)
                            if batch_size > 1:
                                kg_target_index = kg_target_index.expand(batch_size)
                        
                        kg_target_index = kg_target_index.to(scores.device)
                        if (kg_target_index >= num_candidates).any() or (kg_target_index < 0).any():
                            invalid_indices = (kg_target_index >= num_candidates) | (kg_target_index < 0)
                            logger.error(
                                f"Invalid kg_target_index detected! "
                                f"scores.shape={scores.shape}, "
                                f"kg_target_index={kg_target_index}, "
                                f"invalid_mask={invalid_indices}. "
                                f"Clamping to valid range [0, {num_candidates-1}]"
                            )
                            kg_target_index = torch.clamp(kg_target_index, min=0, max=num_candidates - 1)
                        
                        pos_scores = scores.gather(1, kg_target_index.unsqueeze(1)).squeeze(1)  # [B]
                    else:
                        pos_scores = scores[:, 0]  # [B]
                    
                    if torch.isnan(scores).any() or torch.isinf(scores).any():
                        logger.warning("scores contain NaN/Inf before Listwise Softmax, replacing with zeros")
                        scores = torch.nan_to_num(scores, nan=0.0, posinf=10.0, neginf=-10.0)

                    scores = torch.clamp(scores, min=-20.0, max=20.0)
                    scores_max = scores.max(dim=1, keepdim=True)[0]  # [B, 1]
                    scores_stable = scores - scores_max  # [B, N]
                    
                    # Listwise Softmax Loss: L = -log(exp(s_pos) / sum_j(exp(s_j)))
                    # = -log(exp(s_pos)) + log(sum_j(exp(s_j)))
                    # = -s_pos + log(sum_j(exp(s_j)))
                    exp_scores = torch.exp(scores_stable)  # [B, N]
                    sum_exp = exp_scores.sum(dim=1)  # [B]
                    log_sum_exp = torch.log(sum_exp + 1e-10)  # [B]
                    
                    if kg_target_index is not None:
                        pos_scores_stable = scores_stable.gather(1, kg_target_index.unsqueeze(1).to(scores.device)).squeeze(1)  # [B]
                    else:
                        pos_scores_stable = scores_stable[:, 0]  # [B]
                    
                    # Listwise loss: -s_pos + log(sum_j(exp(s_j)))
                    listwise_loss = -pos_scores_stable + log_sum_exp  # [B]

                    # Optional margin regularizer.
                    margin = self.training_config.get("kg_margin", 0.5)
                    margin_lambda = self.training_config.get("kg_margin_lambda", 0.5)

                    if kg_target_index is not None:
                        mask_pos = torch.zeros_like(scores_stable, dtype=torch.bool)
                        mask_pos.scatter_(1, kg_target_index.unsqueeze(1), True)
                    else:
                        mask_pos = torch.zeros_like(scores_stable, dtype=torch.bool)
                        mask_pos[:, 0] = True

                    neg_scores = scores_stable.masked_fill(mask_pos, -1e9)
                    max_neg_scores, _ = neg_scores.max(dim=1)  # [B]

                    # Margin loss: max(0, margin - (s_pos - s_neg))
                    margin_loss = torch.relu(
                        margin - (pos_scores_stable - max_neg_scores)
                    )  # [B]

                    top_k_loss_weight = self.training_config.get("top_k_loss_weight", 0.0)
                    top_k_loss = torch.tensor(0.0, device=scores.device)
                    
                    if top_k_loss_weight > 0:
                        top_k_neg_count = self.training_config.get("top_k_neg_count", 5)
                        top_k_loss_margin = self.training_config.get("top_k_loss_margin", margin)
                        
                        actual_neg_count = num_candidates - 1
                        top_k_neg_count = min(top_k_neg_count, actual_neg_count)
                        
                        if top_k_neg_count > 0:
                            neg_scores_sorted, _ = torch.sort(neg_scores, dim=1, descending=True)  # [B, N]
                            top_k_neg_scores = neg_scores_sorted[:, :top_k_neg_count]  # [B, K]
                            
                            top_k_neg_mean = top_k_neg_scores.mean(dim=1)  # [B]
                            top_k_loss = torch.relu(
                                top_k_loss_margin - (pos_scores_stable - top_k_neg_mean)
                    )  # [B]

                    listwise_weight = self.training_config.get("listwise_weight", 1.0)

                    total_loss_per_sample = listwise_weight * listwise_loss + margin_lambda * margin_loss + top_k_loss_weight * top_k_loss  # [B]
                    
                    # Optional subsampling weights.
                    subsampling_weight = inputs.get("subsampling_weight", None)
                    if subsampling_weight is not None:
                        subsampling_weight = subsampling_weight.to(device)
                        batch_size = scores.size(0)
                        
                        if subsampling_weight.dim() == 0:
                            subsampling_weight = subsampling_weight.unsqueeze(0)
                        
                        if subsampling_weight.dim() == 1:
                            if subsampling_weight.size(0) == 1:
                                subsampling_weight = subsampling_weight.expand(batch_size)
                            elif subsampling_weight.size(0) != batch_size:
                                subsampling_weight = subsampling_weight[0:1].expand(batch_size)
                        
                        total_loss_per_sample = total_loss_per_sample * subsampling_weight  # [B] * [B] = [B]
                    
                    kg_loss = total_loss_per_sample.mean()

                    if torch.isnan(kg_loss) or torch.isinf(kg_loss):
                        logger.warning(
                            f"KG loss NaN/Inf after calculation (listwise_loss mean: {listwise_loss.mean().item():.4f}), "
                            f"fallback to 0.1"
                        )
                        kg_loss = torch.tensor(0.1, device=device, requires_grad=True)

                # =====================================================
                # Combine losses (keep KG dominant by default)
                # =====================================================
                w_kg = float(self.training_config.get("kg_loss_weight", 1.0))
                # Default 0.0 => purely KG-ranking driven training (recommended for KG-completion focus)
                w_lm = float(self.training_config.get("lm_loss_weight", 0.0))

                # If model didn't return lm_loss, treat as 0.
                if lm_loss is None or (isinstance(lm_loss, torch.Tensor) and (torch.isnan(lm_loss) or torch.isinf(lm_loss))):
                    lm_loss_safe = torch.tensor(0.0, device=device)
                else:
                    lm_loss_safe = lm_loss.to(device) if isinstance(lm_loss, torch.Tensor) else torch.tensor(float(lm_loss), device=device)

                loss = w_kg * kg_loss + w_lm * lm_loss_safe

                # Attach components for debugging/metrics consumers
                if isinstance(outputs, dict):
                    outputs["kg_loss"] = kg_loss.detach()
                    outputs["lm_loss"] = lm_loss_safe.detach()
                    outputs["total_loss"] = loss.detach()
                else:
                    # best-effort: add attributes if possible
                    try:
                        setattr(outputs, "kg_loss", kg_loss.detach())
                        setattr(outputs, "lm_loss", lm_loss_safe.detach())
                        setattr(outputs, "total_loss", loss.detach())
                    except Exception:
                        pass

            else:
                raise RuntimeError("No valid KG scores found. Expected kg_scores from model forward pass.")


        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise
            self.logger.error(f"Runtime error during compute_loss: {e}", exc_info=True)

            loss = torch.tensor(0.1, device=device, requires_grad=True)

            outputs = None

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override Trainer.log to keep custom file loggers in sync."""
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)

        if 'loss' in logs:
            self.logger.info(f"Step {self.state.global_step}: loss = {logs['loss']:.4f}")

        if 'learning_rate' in logs:
            self.logger.info(f"Step {self.state.global_step}: lr = {logs['learning_rate']:.2e}")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Evaluation:
        - compute KG ranking metrics first (for early stopping)
        - then compute eval loss (optional monitoring)
        """
        eval_dataset = eval_dataset or self.eval_dataset
        
        max_eval_samples = 2000
        eval_indices = None
        if eval_dataset is not None:
            indices = list(range(len(eval_dataset)))
            if max_eval_samples is not None and max_eval_samples < len(indices):
                random.seed(42)
                eval_indices = random.sample(indices, max_eval_samples)
                eval_indices.sort()
                self.eval_logger.info(
                    f"Random sampling (seed=42): {max_eval_samples} / {len(eval_dataset)} samples for eval"
                )
            else:
                eval_indices = indices
                self.eval_logger.info(
                    f"Evaluating on full validation set: {len(eval_dataset)} samples"
                )
        
        eval_output = {}
        if eval_dataset is not None:
            try:
                self.eval_logger.info("=== KG Ranking Evaluation ===")
                
                was_training = self.model.training
                original_dtype = next(self.model.parameters()).dtype

                data_path = self.training_config.get("data_path", "")
                if data_path and os.path.exists(data_path):
                    def _load_triples(path: str):
                        """
                        Load triples from a file:
                        - optional first line count (single integer) is skipped
                        - each row is h t r, converted to (h, r, t)
                        """
                        triples = []
                        if os.path.exists(path):
                            with open(path, 'r') as f:
                                first_line = True
                                for line in f:
                                    line = line.strip()
                                    if not line:
                                        continue
                                    
                                    parts = line.split()
                                    if first_line and len(parts) == 1:
                                        first_line = False
                                        continue
                                    
                                    if len(parts) >= 3:
                                        try:
                                            h, t, r = map(int, parts[:3])
                                            triples.append((h, r, t))
                                        except ValueError:
                                            continue
                                    first_line = False
                        return triples
                    
                    train_path = os.path.join(data_path, "train2id.txt") if os.path.isdir(data_path) else data_path.replace("valid2id.txt", "train2id.txt")
                    valid_path = os.path.join(data_path, "valid2id.txt") if os.path.isdir(data_path) else data_path
                    test_path = os.path.join(data_path, "test2id.txt") if os.path.isdir(data_path) else data_path.replace("valid2id.txt", "test2id.txt")
                    
                    train_triples = _load_triples(train_path)
                    valid_triples = _load_triples(valid_path)
                    test_triples = _load_triples(test_path)
                    all_true_triples = train_triples + valid_triples + test_triples
                    
                    self.eval_logger.info(
                        f"Loaded all_true_triples: train={len(train_triples)}, valid={len(valid_triples)}, "
                        f"test={len(test_triples)}, total={len(all_true_triples)}"
                    )
                else:
                    all_true_triples = None
                    self.eval_logger.warning("Failed to load all_true_triples; filtered eval may be inconsistent")

                two_stage_eval = bool(self.training_config.get("use_two_stage_eval", True))

                metrics = evaluate_kg_ranking(
                    model=self.model,
                    dataset=eval_dataset,
                    device=self.device,
                    max_eval_samples=max_eval_samples,
                    all_true_triples=all_true_triples,
                    filtered=True,
                    evaluate_heads=False,
                    evaluate_tails=True,
                    two_stage=two_stage_eval,
                    rerank_top_m=int(self.training_config.get("num_negatives", 99)) + 1,
                    kge_embeddings_dir=None,
                    kge_gamma=12.0,
                    kge_embedding_range=2.0,
                    random_seed=42,
                )

                self.eval_logger.info(
                    f"[KG Ranking] "
                    f"MR={metrics['mr']:.2f} | "
                    f"MRR={metrics['mrr']:.4f} | "
                    f"Hit@1={metrics['hit1']:.4f} | "
                    f"Hit@3={metrics['hit3']:.4f} | "
                    f"Hit@10={metrics['hit10']:.4f}"
                )

                eval_output.update({
                    f"{metric_key_prefix}_kg_mr": metrics["mr"],
                    f"{metric_key_prefix}_kg_mrr": metrics["mrr"],
                    f"{metric_key_prefix}_kg_hit1": metrics["hit1"],
                    f"{metric_key_prefix}_kg_hit3": metrics["hit3"],
                    f"{metric_key_prefix}_kg_hit10": metrics["hit10"],
                })
                
                
                torch.cuda.empty_cache()
                gc.collect()
                self.model.train(was_training)
                if next(self.model.parameters()).dtype != original_dtype:
                    self.model = self.model.to(dtype=original_dtype)
                torch.cuda.empty_cache()

            except Exception as e:
                self.eval_logger.warning(f"KG ranking evaluation failed: {e}")
                import traceback
                self.eval_logger.warning(traceback.format_exc())
                eval_output.update({
                    f"{metric_key_prefix}_kg_mr": float("inf"),
                    f"{metric_key_prefix}_kg_mrr": 0.0,
                    f"{metric_key_prefix}_kg_hit1": 0.0,
                    f"{metric_key_prefix}_kg_hit3": 0.0,
                    f"{metric_key_prefix}_kg_hit10": 0.0,
                })
                torch.cuda.empty_cache()
                gc.collect()
                if hasattr(self, 'model'):
                    self.model.train(True)
        
        # Compute eval loss using the same compute_loss() as training.
        try:
            self.model.eval()
            total_loss = 0.0
            num_samples = 0
            
            if eval_dataset is not None and eval_indices is not None:
                eval_subset = Subset(eval_dataset, eval_indices)
                eval_dataloader = DataLoader(
                    eval_subset,
                    batch_size=self.args.per_device_eval_batch_size,
                    collate_fn=collate_fn,
                    shuffle=False,
                    num_workers=0,
                )
            else:
                eval_dataloader = self.get_eval_dataloader()
            
            with torch.no_grad():
                for step, inputs in enumerate(eval_dataloader):
                    actual_batch_size = inputs.get("input_ids", torch.tensor(1)).size(0)
                    
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                    
                    loss = self.compute_loss(self.model, inputs, return_outputs=False)
                    
                    if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item() * actual_batch_size
                        num_samples += actual_batch_size
            
            if num_samples > 0:
                avg_loss = total_loss / num_samples
                eval_output[f"{metric_key_prefix}_loss"] = avg_loss
                self.eval_logger.info(
                    f"[KG Ranking Loss] eval_loss={avg_loss:.4f} "
                    f"(computed on {num_samples} samples)"
                )
            else:
                eval_output[f"{metric_key_prefix}_loss"] = float("inf")
                self.eval_logger.warning("Failed to compute KG ranking loss, using inf")
                
        except Exception as e:
            self.eval_logger.warning(f"KG ranking loss computation failed: {e}")
            import traceback
            self.eval_logger.warning(traceback.format_exc())
            eval_output[f"{metric_key_prefix}_loss"] = float("inf")

        return eval_output


def train_multimodal_adap(
        base_model: str,
        data_path: str,
        output_dir: str,
        training_config: Dict[str, Any]

):
    """Train MultiModalAdap."""
    log_file = MMTConfig.setup_logging(log_dir=os.path.join(output_dir, "logs"))
    logger = MMTConfig.get_trainer_logger()
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Starting MultiModal Adap Training")
    logger.info(f"Log file: {log_file}")

    MMTConfig.setup_cuda()
    device = MMTConfig.get_device("train")
    logger.info(f"Training on device: {device}")
    logger.info(f"Output directory: {output_dir}")

    logger.info("=== Training Configuration ===")
    for key, value in training_config.items():
        if key != 'adapter_config':
            logger.info(f"  {key}: {value}")
    logger.info("=== Adapter Configuration ===")
    for key, value in training_config.get('adapter_config', {}).items():
        logger.info(f"  {key}: {value}")

    memory_optimizer = MemoryOptimizer(strategy="lora")
    model, tokenizer = memory_optimizer.get_optimized_model(base_model)
    
    try:
        if hasattr(model, "llama_model") and hasattr(model.llama_model, "config"):
            model.llama_model.config.use_cache = False
    except Exception:
        pass

    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    tokenizer.padding_side = "left"

    multimodal_model = MultiModalAdap(
        model=model,
        num_prefix=training_config.get("num_prefix", 1),
        kge_model_path=training_config.get("kge_model_path"),
        adapter_config=training_config.get("adapter_config")
    )

    multimodal_model.multimodal_adapter = multimodal_model.multimodal_adapter.to(device)

    train_data_path = os.path.join(data_path, "train2id.txt") if os.path.isdir(data_path) else data_path
    
    from utils.prompter import Prompter
    prompt_template_name = training_config.get("prompt_template", "kg_completion")
    prompter = Prompter(prompt_template_name)
    logger.info(f"Using prompt template: {prompt_template_name}")
    
    use_retriever = training_config.get("use_retriever_for_training", True)
    train_dataset = MMTMultiModalDataset(
        train_data_path,
        tokenizer,
        prompter=prompter,
        max_length=training_config.get("cutoff_len", 256),
        num_negatives=training_config.get("num_negatives", 63),
        use_retriever_for_training=use_retriever,
        kge_embeddings_dir=training_config.get("kge_embeddings_dir", None),
        kge_gamma=training_config.get("kge_gamma", 12.0),
        kge_embedding_range=training_config.get("kge_embedding_range", 2.0),
        device=torch.device("cpu"),
    )
    if use_retriever:
        logger.info("Training candidate selection: multimodal RotatE retriever")
    else:
        logger.info("Training candidate selection: random sampling (retriever disabled)")

    val_dataset = None
    val_data_path = os.path.join(data_path, "valid2id.txt")
    if os.path.exists(val_data_path):
        val_dataset = MMTMultiModalDataset(
            val_data_path,
            tokenizer,
            prompter=prompter,
            max_length=training_config.get("cutoff_len", 256),
            num_negatives=training_config.get("num_negatives", 63),
            use_retriever_for_training=use_retriever,
            kge_embeddings_dir=training_config.get("kge_embeddings_dir", None),
            kge_gamma=training_config.get("kge_gamma", 12.0),
            kge_embedding_range=training_config.get("kge_embedding_range", 2.0),
            device=torch.device("cpu"),
        )
        logger.info(f"Validation set loaded: {len(val_dataset)} samples (use_retriever_for_training={use_retriever})")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get("micro_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        warmup_steps=0,
        num_train_epochs=training_config.get("num_epochs", 3),
        learning_rate=training_config.get("learning_rate", 3e-4),
        lr_scheduler_type="constant",
        fp16=training_config.get("fp16",False),
        logging_steps=training_config.get("logging_steps", 10),
        optim=training_config.get("optim", "adamw_torch"),
        eval_strategy="steps" if val_dataset else "no",
        save_safetensors=False,
        eval_steps=training_config.get("eval_steps", 1000),
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 1000),
        save_total_limit=training_config.get("save_total_limit", 5),
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_kg_mrr"),
        greater_is_better=training_config.get("greater_is_better", True),
        ddp_find_unused_parameters=False,
        group_by_length=training_config.get("group_by_length", False),
        report_to=training_config.get("report_to", []),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        no_cuda=False if torch.cuda.is_available() else True,
        local_rank=-1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_first_step=True,
        logging_strategy="steps",
        resume_from_checkpoint=training_config.get("resume_from_checkpoint", None),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
    )

    trainer = MultiModalTrainer(
        training_config=training_config,
        model=multimodal_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    logger.info(f"Using constant LR: {training_config.get('learning_rate', 5e-5):.2e}")

    if val_dataset is not None:
        early_stop_patience = training_config.get("early_stop_patience", 3)
        metric_name = training_config.get("metric_for_best_model", "eval_kg_mrr")
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stop_patience,
            early_stopping_threshold=0.0,
        )
        trainer.add_callback(early_stopping)
        logger.info(f"Early stopping enabled: patience={early_stop_patience}, metric={metric_name}")

    try:
        trainer.train(resume_from_checkpoint=training_config.get("resume_from_checkpoint", None))
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        try:
            emergency_save_path = os.path.join(output_dir, "emergency_save")
            trainer.save_model(emergency_save_path)
            logger.info(f"Emergency save created at: {emergency_save_path}")
        except Exception as save_error:
            logger.error(f"Failed to create emergency save: {save_error}", exc_info=True)
        raise

    logger.info("=== Training Completed ===")
    
    if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None:
        logger.info("Best model found:")
        logger.info(f"   - Step: {trainer.state.best_global_step}")
        logger.info(f"   - Metric ({training_config.get('metric_for_best_model', 'eval_kg_mrr')}): {trainer.state.best_metric:.6f}")
        logger.info(f"   - Checkpoint: {trainer.state.best_model_checkpoint}")
        
        if training_args.load_best_model_at_end:
            logger.info("Best model has been automatically loaded by Trainer")
    else:
        logger.warning("No best model information found in trainer state")
    
    logger.info("Saving adapters and final model...")
    multimodal_model.save_adapters(output_dir)
    
    trainer.save_model(output_dir)
    logger.info(f"Final model saved to: {output_dir}")
    
    if hasattr(trainer.state, 'best_model_checkpoint') and trainer.state.best_model_checkpoint:
        best_checkpoint = trainer.state.best_model_checkpoint
        best_model_dir = os.path.join(output_dir, "best_model")
        if os.path.exists(best_checkpoint) and not os.path.exists(best_model_dir):
            try:
                import shutil
                logger.info(f"Creating best_model directory from checkpoint: {best_checkpoint}")
                shutil.copytree(best_checkpoint, best_model_dir)
                logger.info(f"Best model copied to: {best_model_dir}")
            except Exception as e:
                logger.warning(f"Failed to create best_model directory: {e}")
    
    return multimodal_model





