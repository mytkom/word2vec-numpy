import re
from typing import List, Tuple, Dict

import numpy as np
import requests


SHAKESPEARE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"


def fetch_shakespeare(url: str = SHAKESPEARE_URL) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text


def tokenize(text: str) -> List[str]:
    text = text.lower()
    # simple word tokenizer
    tokens = re.findall(r"[a-z']+|[.!?]", text)
    return tokens


def build_vocab(tokens: List[str], max_vocab_size: int = 10000) -> Tuple[Dict[str, int], Dict[int, str]]:
    freqs: Dict[str, int] = {}
    for t in tokens:
        freqs[t] = freqs.get(t, 0) + 1
    # sort by frequency
    sorted_tokens = sorted(freqs.items(), key=lambda x: -x[1])[:max_vocab_size - 1]
    word_to_id = {"<UNK>": 0}
    for i, (w, _) in enumerate(sorted_tokens, start=1):
        word_to_id[w] = i
    id_to_word = {i: w for w, i in word_to_id.items()}
    return word_to_id, id_to_word


def encode_tokens(tokens: List[str], word_to_id: Dict[str, int]) -> np.ndarray:
    unk = word_to_id["<UNK>"]
    ids = [word_to_id.get(t, unk) for t in tokens]
    return np.array(ids, dtype=np.int64)


def generate_skipgram_pairs(
    ids: np.ndarray,
    window_size: int,
) -> List[Tuple[int, int]]:
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


def generate_cbow_pairs(
    ids: np.ndarray,
    window_size: int,
) -> List[Tuple[np.ndarray, int]]:
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


def train_val_split(indices: np.ndarray, val_ratio: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(indices))
    cut = int(len(indices) * (1.0 - val_ratio))
    train_idx = perm[:cut]
    val_idx = perm[cut:]
    return train_idx, val_idx

