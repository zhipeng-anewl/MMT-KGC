# MMT-KGC

Multi-Modal Transform for Knowledge Graph Completion.

## Data

Large data files (model weights, embeddings, datasets) are **not stored in this repository** due to GitHub's 25 MB file size limit. Download them separately:

- Embedding files (`*.pth`, `*.pt`): see the releases page or the project's data server.
- Raw knowledge graph triples: available from the [TCM-MMKG](https://github.com/zhipeng-anewl/MMT-KGC/tree/data) data branch (small files only).

Place downloaded files under `tcm-mmkg/embeddings/` before running experiments.

## Large File Handling

This repository uses `.gitignore` to exclude binary model files and large datasets. If you need to version-control large files, use [Git LFS](https://git-lfs.com/):

```bash
git lfs install
git lfs track "*.pth" "*.pt" "*.bin"
git add .gitattributes
```

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python train.py --config configs/default.yaml
```
