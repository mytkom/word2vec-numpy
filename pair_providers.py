from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterable

import numpy as np

from data_utils import train_val_split


@dataclass
class BasePairProvider:
    ids: np.ndarray
    window_size: int
    val_ratio: float
    seed: int = 123
    _pairs: Optional[List] = field(default=None, init=False, repr=False)
    _train_pairs: Optional[List] = field(default=None, init=False, repr=False)
    _val_pairs: Optional[List] = field(default=None, init=False, repr=False)

    def generate_pairs(self) -> List:
        raise NotImplementedError

    def get_train_val_pairs(self) -> Tuple[List, List]:
        if self._train_pairs is None or self._val_pairs is None:
            if self._pairs is None:
                self._pairs = self.generate_pairs()
            indices = np.arange(len(self._pairs))
            train_idx, val_idx = train_val_split(indices, val_ratio=self.val_ratio, seed=self.seed)
            self._train_pairs = [self._pairs[i] for i in train_idx]
            self._val_pairs = [self._pairs[i] for i in val_idx]
        return self._train_pairs, self._val_pairs

    @staticmethod
    def batch(pairs: List, batch_size: int, seed: int = 123) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError


class SkipGramPairProvider(BasePairProvider):
    def generate_pairs(self) -> List[Tuple[int, int]]:
        ids = self.ids
        window_size = self.window_size
        pairs: List[Tuple[int, int]] = []
        n = len(ids)
        for i in range(n):
            center = ids[i]
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            for j in range(left, right):
                if j == i:
                    continue
                context = ids[j]
                pairs.append((center, context))
        return pairs

    @staticmethod
    def batch(
        pairs: List[Tuple[int, int]],
        batch_size: int,
        seed: int = 123,
    ):
        rng = np.random.RandomState(seed)
        indices = np.arange(len(pairs))
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            centers = np.array([pairs[i][0] for i in idx], dtype=np.int64)
            contexts = np.array([pairs[i][1] for i in idx], dtype=np.int64)
            yield centers, contexts


class CBOWPairProvider(BasePairProvider):
    def generate_pairs(self) -> List[Tuple[np.ndarray, int]]:
        ids = self.ids
        window_size = self.window_size
        pairs: List[Tuple[np.ndarray, int]] = []
        n = len(ids)
        for i in range(n):
            left = max(0, i - window_size)
            right = min(n, i + window_size + 1)
            context_indices = [j for j in range(left, right) if j != i]
            if not context_indices:
                continue
            context = ids[context_indices]
            target = ids[i]
            pairs.append((context, target))
        return pairs

    @staticmethod
    def batch(
        pairs: List[Tuple[np.ndarray, int]],
        batch_size: int,
        seed: int = 123,
    ):
        rng = np.random.RandomState(seed)
        indices = np.arange(len(pairs))
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            contexts = []
            targets = []
            max_len = 0
            for i in idx:
                ctx_ids, tgt = pairs[i]
                contexts.append(ctx_ids)
                targets.append(tgt)
                if len(ctx_ids) > max_len:
                    max_len = len(ctx_ids)
            padded = np.zeros((len(idx), max_len), dtype=np.int64)
            for k, ctx_ids in enumerate(contexts):
                padded[k, : len(ctx_ids)] = ctx_ids
            yield padded, np.array(targets, dtype=np.int64)

