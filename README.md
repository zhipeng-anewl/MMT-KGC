# MMT-KGC

MMT-KGC is a multimodal TCM knowledge graph completion project.

This repository supports a full training pipeline:
1. Train structural KGE embeddings
2. Train a multimodal KGE reranker (coarse retriever)
3. Train the main MMT-KGC model

The project is prepared for open-source reproducibility with local, repository-relative paths.

## Requirements

- Python 3.9.12
- PyTorch >= 2.0.0
- transformers >= 4.30.0
- peft >= 0.5.0
- numpy >= 1.21.0
- tqdm >= 4.64.0

Recommended install:

```bash
pip install "torch>=2.0.0" "transformers>=4.30.0" "peft>=0.5.0" "numpy>=1.21.0" "tqdm>=4.64.0"
```

## dataset

- Unified data root: data/
- Unified multimodal features: data/embeddings/visual.pth, textual.pth, numeric.pth
- Triple parsing unified to your actual processed format

## Base LLM

Default model expected by the project:
- llm/llama-2-7b
The repository does not include large LLM weights.
Download your preferred model and place it under llm/.
You can override the model path with command arguments.

## Reproducible Training Pipeline

Run from project root.

### Step 1: Train Structural KGE Embeddings

```bash
python training/train_kge.py --model RotatE 
```

Output embeddings are saved to data/embeddings/.

### Step 2: Train Multimodal KGE Coarse Retriever

```bash
python training/train_multimodal_kge.py --model RotatE --fusion_method 
```

Output checkpoints are saved to data/multimodal_kge_models/.

### Step 3: Train Main Multimodal Adap Model

```bash
python start_training.py --model_size 7b --cuda  
```

## Final Evaluation

```bash
python evaluate.py --cuda
```

## Citation

If this repository is useful for your work, please cite your project or paper accordingly.