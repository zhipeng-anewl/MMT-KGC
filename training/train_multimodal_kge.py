#!/usr/bin/env python3
"""Train a multimodal KGE model (e.g., RotatE) with unified multimodal features."""

import os

import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJ_ROOT)

from utils.config import MMTConfig
from models.multimodal_kge_model import MultimodalKGEModel
from training.dataloader import TrainDataset, TestDataset, BidirectionalOneShotIterator


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
    parser = argparse.ArgumentParser(description='Train a multimodal KGE model')
    parser.add_argument('--model', default='TransE', choices=['TransE','DistMult','ComplEx','RotatE','pRotatE'])
    parser.add_argument('--data_path', default=MMTConfig.DATA_PATHS['processed'])
    parser.add_argument('--log_dir',   default=f'{PROJ_ROOT}/data/multimodal_kge_logs')
    parser.add_argument('--emb_dir',   default=MMTConfig.DATA_PATHS['embeddings'],
                       help='Multimodal feature directory (visual/textual/numeric)')
    parser.add_argument('--save_dir',  default=f'{PROJ_ROOT}/data/multimodal_kge_models',
                       help='Checkpoint directory (stores .pt and .npy)')
    parser.add_argument('--dim',       default=64,   type=int)
    parser.add_argument('--gamma',     default=12,    type=float)
    parser.add_argument('--lr',        default=5e-4,  type=float)
    parser.add_argument('--batch',     default=1024,  type=int)
    parser.add_argument('--neg_size',  default=256,   type=int, 
                       help='Negative samples per positive during training')
    parser.add_argument('--test_batch',default=64,    type=int)
    parser.add_argument('--max_step',  default=100000,type=int)
    parser.add_argument('--test_step', default=2000, type=int)
    parser.add_argument('--cpu_num',   default=4,     type=int,
                       help='DataLoader workers')
    parser.add_argument('--cuda',      action='store_true',default=True)
    parser.add_argument('--cuda_devices', default=None, type=str,
                       help='Optional: set CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1"')
    parser.add_argument('--double_e',  action='store_true')
    parser.add_argument('--double_r',  action='store_true')
    
    parser.add_argument('--fusion_method', default='add', choices=['concat', 'add', 'weighted'],
                       help='Fusion method')
    parser.add_argument('--multimodal_weight', default=0.5, type=float,
                       help='Multimodal weight (only for weighted fusion)')
    
    parser.add_argument('--early_stop_patience', default=3, type=int,
                       help='Early stop patience after saving the best model')
    
    return parser.parse_args()


def test_step(model, test_triples, all_true_triples, args):
    model.eval()
    test_dataset_head = TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'head-batch')
    test_dataset_tail = TestDataset(test_triples, all_true_triples, args.nentity, args.nrelation, 'tail-batch')
    dataloader_head = DataLoader(test_dataset_head, batch_size=args.test_batch,
                                 num_workers=args.cpu_num, collate_fn=TestDataset.collate_fn)
    dataloader_tail = DataLoader(test_dataset_tail, batch_size=args.test_batch,
                                 num_workers=args.cpu_num, collate_fn=TestDataset.collate_fn)

    logs = []
    with torch.no_grad():
        for dataloader in (dataloader_head, dataloader_tail):
            for positive, negative, filter_bias, mode in dataloader:
                if args.cuda:
                    positive, negative, filter_bias = positive.cuda(), negative.cuda(), filter_bias.cuda()
                score = model((positive, negative), mode=mode)
                score += filter_bias
                argsort = torch.argsort(score, dim=1, descending=True)
                rank = (argsort == (positive[:, 0] if mode=='head-batch' else positive[:, 2]).view(-1,1)).nonzero(as_tuple=False)[:,1] + 1
                for r in rank.cpu().numpy():
                    logs.append({'MRR':1/r, 'MR':r, 'HITS@1':r<=1, 'HITS@3':r<=3, 'HITS@10':r<=10})
    metrics = {k:np.mean([lg[k] for lg in logs]) for k in logs[0]}
    return metrics

def main():
    args = parse_args()
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.emb_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    args.nentity = _read_count_first_line(f'{args.data_path}/entity2id.txt')
    args.nrelation = _read_count_first_line(f'{args.data_path}/relation2id.txt')

    train_triples = _load_triples_ht_r(f'{args.data_path}/train2id.txt')
    valid_triples = _load_triples_ht_r(f'{args.data_path}/valid2id.txt')
    test_triples = _load_triples_ht_r(f'{args.data_path}/test2id.txt')
    all_true      = [tuple(t) for t in np.vstack([train_triples, valid_triples, test_triples])]

    log_file = f'{args.log_dir}/{args.model}_{args.dim}d_multimodal_{args.fusion_method}.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    
    logging.info('=' * 70)
    logging.info(f'Start training multimodal KGE: model={args.model}')
    logging.info(f'fusion_method={args.fusion_method}')
    logging.info(f'multimodal_weight={args.multimodal_weight}')
    logging.info('=' * 70)

    model = MultimodalKGEModel(
        model_name=args.model,
        nentity=args.nentity,
        nrelation=args.nrelation,
        hidden_dim=args.dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_e,
        double_relation_embedding=args.double_r,
        multimodal_data_dir=args.emb_dir,
        fusion_method=args.fusion_method,
        multimodal_weight=args.multimodal_weight
    )
    if args.cuda:
        model = model.cuda()
        if hasattr(model, '_move_multimodal_data_to_device'):
            model._move_multimodal_data_to_device(model.entity_embedding.device)
        if torch.cuda.is_available():
            logging.info(f'Model moved to GPU: {torch.cuda.get_device_name(0)}')
        else:
            logging.warning('CUDA is not available. Running on CPU.')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logging.info('Initializing dataloaders...')
    train_dataset_head = TrainDataset(train_triples, args.nentity, args.nrelation,
                                      negative_sample_size=args.neg_size, mode='head-batch')
    train_dataset_tail = TrainDataset(train_triples, args.nentity, args.nrelation,
                                      negative_sample_size=args.neg_size, mode='tail-batch')
    
    logging.info(f'Using {args.cpu_num} DataLoader workers')
    train_loader_head = DataLoader(train_dataset_head, batch_size=args.batch,
                                   shuffle=True, num_workers=args.cpu_num,
                                   collate_fn=TrainDataset.collate_fn, 
                                   persistent_workers=True if args.cpu_num > 0 else False)
    train_loader_tail = DataLoader(train_dataset_tail, batch_size=args.batch,
                                   shuffle=True, num_workers=args.cpu_num,
                                   collate_fn=TrainDataset.collate_fn,
                                   persistent_workers=True if args.cpu_num > 0 else False)
    train_iterator = BidirectionalOneShotIterator(train_loader_head, train_loader_tail)
    logging.info('Dataloaders ready. Start training...')

    best_mrr = 0.0
    best_step = 0
    no_improve_count = 0
    early_stopped = False
    
    for step in range(1, args.max_step + 1):
        model.train()
        try:
            positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)
        except StopIteration:
            train_iterator = BidirectionalOneShotIterator(train_loader_head, train_loader_tail)
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
            logging.info(f'Valid  MR={metrics["MR"]:.2f}  MRR={metrics["MRR"]:.4f}  '
                         f'HITS@1={metrics["HITS@1"]:.4f}  HITS@3={metrics["HITS@3"]:.4f}  HITS@10={metrics["HITS@10"]:.4f}')
            if metrics['MRR'] > best_mrr:
                best_mrr = metrics['MRR']
                best_step = step
                no_improve_count = 0
                suffix = f'multimodal_{args.fusion_method}'
                
                model_state = {
                    'entity_embedding': model.entity_embedding.detach().cpu(),  # nn.Parameter -> Tensor
                    'relation_embedding': model.relation_embedding.detach().cpu(),
                    'model_name': args.model,
                    'nentity': args.nentity,
                    'nrelation': args.nrelation,
                    'dim': args.dim,
                    'gamma': args.gamma,
                    'double_entity_embedding': args.double_e,
                    'double_relation_embedding': args.double_r,
                    'fusion_method': args.fusion_method,
                    'multimodal_weight': args.multimodal_weight,
                    'step': step,
                    'mrr': best_mrr,
                }
                
                if hasattr(model, 'herb_image_proj') and model.herb_image_proj is not None:
                    model_state['herb_image_proj'] = model.herb_image_proj.state_dict()
                if hasattr(model, 'mol_graph_proj') and model.mol_graph_proj is not None:
                    model_state['mol_graph_proj'] = model.mol_graph_proj.state_dict()
                if hasattr(model, 'mol_attr_proj') and model.mol_attr_proj is not None:
                    model_state['mol_attr_proj'] = model.mol_attr_proj.state_dict()
                
                model_file = f'{args.save_dir}/{args.model}_best_model_{args.dim}d_{suffix}.pt'
                torch.save(model_state, model_file)
                logging.info(f'Saved checkpoint: {model_file} (step {step}, MRR={best_mrr:.4f})')
                
                ent_emb = model.entity_embedding.detach().cpu().numpy()
                rel_emb = model.relation_embedding.detach().cpu().numpy()
                np.save(f'{args.save_dir}/{args.model}_entity_{args.dim}d_{suffix}.npy', ent_emb)
                np.save(f'{args.save_dir}/{args.model}_relation_{args.dim}d_{suffix}.npy', rel_emb)
                logging.info('Also saved .npy embeddings for compatibility')
            else:
                if best_step > 0:
                    no_improve_count += 1
                    logging.info(
                        f'Validation MRR did not improve (current={metrics["MRR"]:.4f}, best={best_mrr:.4f}). '
                        f'No-improve count: {no_improve_count}/{args.early_stop_patience}'
                    )
                    
                    if no_improve_count >= args.early_stop_patience:
                        logging.info(f'Early stop: no improvement for {args.early_stop_patience} evals after best checkpoint.')
                        logging.info(f'Best checkpoint: step={best_step}, MRR={best_mrr:.4f}')
                        early_stopped = True
                        break

    if early_stopped:
        logging.info('Training stopped early.')
    else:
        logging.info(f'Training finished (max_step={args.max_step}).')
    
    if best_step == 0:
        logging.warning('No best checkpoint found. Skip test evaluation.')
        return
    
    logging.info(f'Loading best checkpoint (step {best_step}, valid MRR={best_mrr:.4f}) for test evaluation...')
    suffix = f'multimodal_{args.fusion_method}'
    model_file = f'{args.save_dir}/{args.model}_best_model_{args.dim}d_{suffix}.pt'
    
    if os.path.exists(model_file):
        logging.info(f'Loading full checkpoint from .pt: {model_file}')
        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
        with torch.no_grad():
            model.entity_embedding.data.copy_(checkpoint['entity_embedding'])
            model.relation_embedding.data.copy_(checkpoint['relation_embedding'])
        
        if 'herb_image_proj' in checkpoint and hasattr(model, 'herb_image_proj') and model.herb_image_proj is not None:
            model.herb_image_proj.load_state_dict(checkpoint['herb_image_proj'])
        if 'mol_graph_proj' in checkpoint and hasattr(model, 'mol_graph_proj') and model.mol_graph_proj is not None:
            model.mol_graph_proj.load_state_dict(checkpoint['mol_graph_proj'])
        if 'mol_attr_proj' in checkpoint and hasattr(model, 'mol_attr_proj') and model.mol_attr_proj is not None:
            model.mol_attr_proj.load_state_dict(checkpoint['mol_attr_proj'])
        
        logging.info('Checkpoint loaded (including projection layers)')

        if args.cuda:
            model = model.cuda()
            if hasattr(model, '_move_multimodal_data_to_device'):
                model._move_multimodal_data_to_device(model.entity_embedding.device)
    else:
        logging.warning(f'.pt checkpoint not found; loading from .npy (projection layers not included): {model_file}')
        best_ent_emb = np.load(f'{args.save_dir}/{args.model}_entity_{args.dim}d_{suffix}.npy')
        best_rel_emb = np.load(f'{args.save_dir}/{args.model}_relation_{args.dim}d_{suffix}.npy')
        with torch.no_grad():
            model.entity_embedding.data = torch.from_numpy(best_ent_emb).float()
            model.relation_embedding.data = torch.from_numpy(best_rel_emb).float()

        if args.cuda:
            model = model.cuda()
            if hasattr(model, '_move_multimodal_data_to_device'):
                model._move_multimodal_data_to_device(model.entity_embedding.device)
    
    metrics = test_step(model, test_triples, all_true, args)
    logging.info(f'Test  MR={metrics["MR"]:.2f}  MRR={metrics["MRR"]:.4f}  '
                 f'HITS@1={metrics["HITS@1"]:.4f}  HITS@3={metrics["HITS@3"]:.4f}  HITS@10={metrics["HITS@10"]:.4f}')

if __name__ == '__main__':
    main()

