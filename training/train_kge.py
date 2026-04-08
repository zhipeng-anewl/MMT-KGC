#!/usr/bin/env python3
"""Train a (structural) KGE model with relation-aware negative sampling."""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import sys

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJ_ROOT)

from models.kge_model import KGEModel
from training.dataloader import TestDataset
from training.relation_aware_dataloader import (
    RelationAwareTrainDataset, RelationAwareBidirectionalOneShotIterator
)


def _read_count_first_line(path: str) -> int:
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline().strip()
    return int(first)


def _load_triples_ht_r(path: str):
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
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
                h, t, r = map(int, parts[:3])
                triples.append((h, r, t))
            first_line = False
    return triples


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='RotatE', choices=['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE'])
    parser.add_argument('--data_path', default=f'{PROJ_ROOT}/data/processed')
    parser.add_argument('--log_dir', default=f'{PROJ_ROOT}/data/kge_logs')
    parser.add_argument('--emb_dir', default=f'{PROJ_ROOT}/data/embeddings')

    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--gamma', default=12, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--batch', default=1024, type=int)
    parser.add_argument('--neg_size', default=512, type=int)
    parser.add_argument('--test_batch', default=64, type=int)
    parser.add_argument('--max_step', default=150000, type=int)
    parser.add_argument('--test_step', default=10000, type=int)
    parser.add_argument('--cpu_num', default=4, type=int)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--double_e', action='store_true')
    parser.add_argument('--double_r', action='store_true')

    parser.add_argument('--relation_aware', action='store_true', default=True,
                        help='Enable relation-aware negative sampling')
    parser.add_argument('--custom_weights', action='store_true', default=False,
                        help='Use custom relation weights (if provided by the caller)')

    return parser.parse_args()


def test_step(model, test_triples, all_true_triples, args):
    """Evaluate via filtered ranking and report MR/MRR/Hits@K."""
    model.eval()
    test_dataset_head = TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'head-batch')
    test_dataset_tail = TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'tail-batch')
    dataloader_head = DataLoader(test_dataset_head, batch_size=args.test_batch,
                                 num_workers=0, collate_fn=TestDataset.collate_fn)
    dataloader_tail = DataLoader(test_dataset_tail, batch_size=args.test_batch,
                                 num_workers=0, collate_fn=TestDataset.collate_fn)

    logs = []
    with torch.no_grad():
        for dataloader in (dataloader_head, dataloader_tail):
            for positive, negative, filter_bias, mode in dataloader:
                if args.cuda:
                    positive, negative, filter_bias = positive.cuda(), negative.cuda(), filter_bias.cuda()
                score = model((positive, negative), mode=mode)
                score += filter_bias
                argsort = torch.argsort(score, dim=1, descending=True)
                rank = (argsort == (positive[:, 0] if mode == 'head-batch' else positive[:, 2]).view(-1, 1)).nonzero(
                    as_tuple=False)[:, 1] + 1
                for r in rank.cpu().numpy():
                    logs.append({'MRR': 1 / r, 'MR': r, 'HITS@1': r <= 1, 'HITS@3': r <= 3, 'HITS@10': r <= 10})
    metrics = {k: np.mean([lg[k] for lg in logs]) for k in logs[0]}
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.emb_dir, exist_ok=True)

    args.nentity = _read_count_first_line(f'{args.data_path}/entity2id.txt')
    args.nrelation = _read_count_first_line(f'{args.data_path}/relation2id.txt')

    train_triples = _load_triples_ht_r(f'{args.data_path}/train2id.txt')
    valid_triples = _load_triples_ht_r(f'{args.data_path}/valid2id.txt')
    test_triples = _load_triples_ht_r(f'{args.data_path}/test2id.txt')
    all_true = [tuple(t) for t in np.vstack([train_triples, valid_triples, test_triples])]

    log_file = f'{args.log_dir}/{args.model}_{args.dim}d_relation_aware_secondtimes.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    logging.info(f"Start training (relation-aware negative sampling): model={args.model}")

    model = KGEModel(model_name=args.model,
                     nentity=args.nentity,
                     nrelation=args.nrelation,
                     hidden_dim=args.dim,
                     gamma=args.gamma,
                     double_entity_embedding=args.double_e,
                     double_relation_embedding=args.double_r)
    if args.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    mmt_relations = {
        0: {'name': 'CONTAINS', 'weight': 1.0, 'strategy': 'random'},
        1: {'name': 'HAS_ADME', 'weight': 1.0, 'strategy': 'random'},
        2: {'name': 'INDICATED_FOR', 'weight': 1.6, 'strategy': 'hard'},
        3: {'name': 'INVOLVED_IN', 'weight': 1.3, 'strategy': 'random'},
        4: {'name': 'TARGETS', 'weight': 1.6, 'strategy': 'hard'},
        5: {'name': 'TREATS', 'weight': 2.0, 'strategy': 'hard'},
    }

    relation_weights = {rel_id: info['weight'] for rel_id, info in mmt_relations.items()}
    tcm_relation_config = {
        rel_id: {
            'neg_size_multiplier': info['weight'],
            'sampling_strategy': info['strategy']
        }
        for rel_id, info in mmt_relations.items()
    }

    logging.info(f"Relation config: {mmt_relations}")

    train_dataset_head = RelationAwareTrainDataset(
        train_triples, args.nentity, args.nrelation,
        negative_sample_size=args.neg_size,
        mode='head-batch',
        relation_weights=relation_weights,
        tcm_relation_config=tcm_relation_config
    )

    train_dataset_tail = RelationAwareTrainDataset(
        train_triples, args.nentity, args.nrelation,
        negative_sample_size=args.neg_size,
        mode='tail-batch',
        relation_weights=relation_weights,
        tcm_relation_config=tcm_relation_config
    )

    train_loader_head = DataLoader(train_dataset_head, batch_size=args.batch,
                                   shuffle=True, num_workers=args.cpu_num,
                                   collate_fn=RelationAwareTrainDataset.collate_fn)
    train_loader_tail = DataLoader(train_dataset_tail, batch_size=args.batch,
                                   shuffle=True, num_workers=args.cpu_num,
                                   collate_fn=RelationAwareTrainDataset.collate_fn)

    train_iterator = RelationAwareBidirectionalOneShotIterator(train_loader_head, train_loader_tail)

    best_mrr = 0.0
    best_step = 0
    for step in range(1, args.max_step + 1):
        model.train()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        positive_score = model(positive_sample)

        positive_loss = -F.logsigmoid(positive_score).squeeze(dim=1) * subsampling_weight
        negative_loss = -F.logsigmoid(-negative_score).mean(dim=1) * subsampling_weight
        loss = (positive_loss + negative_loss).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            logging.info(f'Step {step}  loss={loss.item():.4f}')

        if step % args.test_step == 0 or step == args.max_step:
            metrics = test_step(model, valid_triples, all_true, args)
            logging.info(f'Valid Step{step}: MR={metrics["MR"]:.2f} MRR={metrics["MRR"]:.4f} '
                         f'HITS@1={metrics["HITS@1"]:.4f} HITS@3={metrics["HITS@3"]:.4f} HITS@10={metrics["HITS@10"]:.4f}')

            if metrics['MRR'] > best_mrr:
                best_mrr = metrics['MRR']
                best_step = step
                ent_emb = model.entity_embedding.detach().cpu().numpy()
                rel_emb = model.relation_embedding.detach().cpu().numpy()
                np.save(f'{args.emb_dir}/{args.model}_entity_{args.dim}d.npy', ent_emb)
                np.save(f'{args.emb_dir}/{args.model}_relation_{args.dim}d.npy', rel_emb)
                logging.info(f'Saved best embeddings: step={step}, MRR={best_mrr:.4f}')

    logging.info(f'Load best embeddings (step {best_step}, valid MRR={best_mrr:.4f}) for test evaluation...')
    best_ent_emb = np.load(f'{args.emb_dir}/{args.model}_entity_{args.dim}d.npy')
    best_rel_emb = np.load(f'{args.emb_dir}/{args.model}_relation_{args.dim}d.npy')
    with torch.no_grad():
        model.entity_embedding.data = torch.from_numpy(best_ent_emb).float()
        model.relation_embedding.data = torch.from_numpy(best_rel_emb).float()
    if args.cuda:
        model.entity_embedding.data = model.entity_embedding.data.cuda()
        model.relation_embedding.data = model.relation_embedding.data.cuda()
    
    final_metrics = test_step(model, test_triples, all_true, args)
    logging.info(f'Test: MR={final_metrics["MR"]:.2f} MRR={final_metrics["MRR"]:.4f} '
                 f'HITS@1={final_metrics["HITS@1"]:.4f} HITS@3={final_metrics["HITS@3"]:.4f} HITS@10={final_metrics["HITS@10"]:.4f}')


if __name__ == '__main__':
    main()