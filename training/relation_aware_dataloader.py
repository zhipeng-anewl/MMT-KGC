#!/usr/bin/env python3
"""Relation-aware negative sampling dataloader (TCM KG oriented)."""

import numpy as np
import torch
from torch.utils.data import Dataset


class RelationAwareTrainDataset(Dataset):
    """Training dataset with relation-aware negative sampling."""

    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode,
                 relation_weights=None, tcm_relation_config=None):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode

        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

        self.relation_weights = relation_weights or self.calculate_default_relation_weights()
        self.tcm_relation_config = tcm_relation_config or self.get_default_tcm_config()

        print("Relation weights:", self.relation_weights)
        print("Relation config:", self.tcm_relation_config)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]
        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        relation_config = self.tcm_relation_config.get(relation, {})
        negative_sample = self.relation_aware_negative_sampling(
            head, relation, tail, self.mode, relation_config
        )

        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor(positive_sample)

        # Ensure a fixed number of negatives per sample
        if len(negative_sample) != self.negative_sample_size:
            if len(negative_sample) > self.negative_sample_size:
                negative_sample = negative_sample[:self.negative_sample_size]
            else:
                padding_size = self.negative_sample_size - len(negative_sample)
                padding = torch.full((padding_size,), negative_sample[0], dtype=torch.long)
                negative_sample = torch.cat([negative_sample, padding])

        return positive_sample, negative_sample, subsampling_weight, self.mode

    def relation_aware_negative_sampling(self, head, relation, tail, mode, relation_config):
        """Relation-aware negative sampling."""
        neg_size_multiplier = relation_config.get('neg_size_multiplier', 1.0)
        sampling_strategy = relation_config.get('sampling_strategy', 'random')
        entity_constraints = relation_config.get('entity_constraints', None)

        dynamic_neg_size = int(self.negative_sample_size * neg_size_multiplier)
        dynamic_neg_size = max(1, min(dynamic_neg_size, self.negative_sample_size * 3))

        if sampling_strategy == 'domain_constrained':
            negative_sample = self.domain_constrained_sampling(
                head, relation, tail, mode, dynamic_neg_size, entity_constraints
            )
        elif sampling_strategy == 'hard_sampling':
            negative_sample = self.hard_sampling(head, relation, tail, mode, dynamic_neg_size)
        else:  # random
            negative_sample = self.improved_random_sampling(
                head, relation, tail, mode, dynamic_neg_size
            )

        return negative_sample

    def domain_constrained_sampling(self, head, relation, tail, mode, size, constraints):
        """Domain constrained sampling: restrict negatives to a plausible pool."""
        if mode == 'head-batch':
            true_entities = self.true_head.get((relation, tail), [])
            candidate_pool = self.get_constrained_candidates(relation, 'head', constraints)
        else:  # tail-batch
            true_entities = self.true_tail.get((head, relation), [])
            candidate_pool = self.get_constrained_candidates(relation, 'tail', constraints)

        candidate_pool = [e for e in candidate_pool if e not in true_entities]

        if len(candidate_pool) == 0:
            return self.improved_random_sampling(head, relation, tail, mode, size)

        if len(candidate_pool) <= size:
            return np.array(candidate_pool)
        else:
            return np.random.choice(candidate_pool, size=size, replace=False)

    def get_constrained_candidates(self, relation, entity_type, constraints):
        """Return a constrained candidate pool. (Currently a placeholder implementation.)"""
        if constraints is None:
            return list(range(self.nentity))

        return list(range(self.nentity))

    def hard_sampling(self, head, relation, tail, mode, size):
        """Hard negative sampling based on entity frequency."""
        random_negatives = self.improved_random_sampling(head, relation, tail, mode, size * 2)

        if len(random_negatives) == 0:
            return random_negatives

        entity_freq = self._get_entity_frequencies()
        difficulties = []

        for entity in random_negatives:
            freq = entity_freq.get(entity, 1)
            difficulties.append(freq)

        difficulties = np.array(difficulties)
        hard_indices = np.argsort(difficulties)[-size:]

        return random_negatives[hard_indices]

    def improved_random_sampling(self, head, relation, tail, mode, size):
        """Random sampling with filtering against known true entities."""
        if mode == 'head-batch':
            true_entities = self.true_head.get((relation, tail), [])
        else:  # tail-batch
            true_entities = self.true_tail.get((head, relation), [])

        n_true = len(true_entities)
        available_size = self.nentity - n_true

        if available_size <= 0:
            return np.array([], dtype=np.int64)

        size = min(size, available_size)
        all_entities = np.arange(self.nentity)
        mask = np.isin(all_entities, true_entities, invert=True)
        available_entities = all_entities[mask]

        if len(available_entities) <= size:
            return available_entities
        else:
            return np.random.choice(available_entities, size=size, replace=False)

    def calculate_default_relation_weights(self):
        """Compute inverse-frequency relation weights (normalized)."""
        relation_counts = {}
        for _, relation, _ in self.triples:
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

        total = len(self.triples)
        relation_weights = {rel: total / count for rel, count in relation_counts.items()}

        min_w = min(relation_weights.values())
        max_w = max(relation_weights.values())
        relation_weights = {rel: 0.5 + 1.5 * (w - min_w) / (max_w - min_w)
                            for rel, w in relation_weights.items()}

        return relation_weights

    def get_default_tcm_config(self):
        """Default relation config used by this dataset."""
        return {
            0: {'neg_size_multiplier': 1.0, 'sampling_strategy': 'random'},
            1: {'neg_size_multiplier': 1.5, 'sampling_strategy': 'hard_sampling'},
            2: {'neg_size_multiplier': 1.8, 'sampling_strategy': 'hard_sampling'},
            3: {'neg_size_multiplier': 1.2, 'sampling_strategy': 'random'},
            4: {'neg_size_multiplier': 1.6, 'sampling_strategy': 'hard_sampling'},
            5: {'neg_size_multiplier': 2.0, 'sampling_strategy': 'hard_sampling'}
        }

    def get_relation_name(self, relation_id):
        """Convert relation id to a readable name."""
        relation_names = {
            0: 'CONTAINS',
            1: 'HAS_ADME',
            2: 'INDICATED_FOR',
            3: 'INVOLVED_IN',
            4: 'TARGETS',
            5: 'TREATS'
        }
        return relation_names.get(relation_id, f'REL_{relation_id}')

    def _get_entity_frequencies(self):
        """Compute entity frequencies (cached)."""
        if not hasattr(self, '_entity_freq_cache'):
            entity_freq = {}
            for head, relation, tail in self.triples:
                entity_freq[head] = entity_freq.get(head, 0) + 1
                entity_freq[tail] = entity_freq.get(tail, 0) + 1
            self._entity_freq_cache = entity_freq
        return self._entity_freq_cache

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1
            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        true_head = {}
        true_tail = {}
        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)
        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))
        return true_head, true_tail


class RelationAwareBidirectionalOneShotIterator(object):
    """Bidirectional iterator for relation-aware datasets."""

    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data