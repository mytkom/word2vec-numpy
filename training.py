from typing import List, Tuple, Dict, Any
import logging

import numpy as np

from models import SkipGram, CBOW


def batch_skipgram(
    pairs: List[Tuple[int, int]],
    batch_size: int,
    seed: int = 123,
):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        c = np.array([pairs[i][0] for i in idx], dtype=np.int64)
        ctx = np.array([pairs[i][1] for i in idx], dtype=np.int64)
        yield c, ctx


def batch_cbow(
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
        # pad with target itself (won't be used if window>0), or zeros
        padded = np.zeros((len(idx), max_len), dtype=np.int64)
        for k, ctx_ids in enumerate(contexts):
            padded[k, : len(ctx_ids)] = ctx_ids
        yield padded, np.array(targets, dtype=np.int64)


def train_model(
    model_type: str,
    vocab_size: int,
    embedding_dim: int,
    train_pairs,
    val_pairs,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int = 123,
) -> Dict[str, Any]:
    if model_type == "skipgram":
        model = SkipGram(vocab_size, embedding_dim, seed=seed)
        batcher = batch_skipgram
    elif model_type == "cbow":
        model = CBOW(vocab_size, embedding_dim, seed=seed)
        batcher = batch_cbow
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # train
        train_losses = []
        for xb, yb in batcher(train_pairs, batch_size=batch_size, seed=seed + epoch):
            if model_type == "skipgram":
                loss, cache = model.forward(xb, yb)
            else:
                loss, cache = model.forward(xb, yb)
            model.backward(cache, lr)
            train_losses.append(loss)
        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")

        # val
        val_losses = []
        for xb, yb in batcher(val_pairs, batch_size=batch_size, seed=seed + 999 + epoch):
            if model_type == "skipgram":
                loss, _ = model.forward(xb, yb)
            else:
                loss, _ = model.forward(xb, yb)
            val_losses.append(loss)
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)

    return {"model": model, "history": history}


def grid_search(
    vocab_size: int,
    train_pairs_sg,
    val_pairs_sg,
    train_pairs_cbow,
    val_pairs_cbow,
    embedding_dims,
    lrs,
    batch_sizes,
    epochs: int,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for model_type in ["skipgram", "cbow"]:
        for d in embedding_dims:
            for lr in lrs:
                for bs in batch_sizes:
                    logging.info(f"Training {model_type} with dim={d}, lr={lr}, bs={bs}")
                    if model_type == "skipgram":
                        train_pairs = train_pairs_sg
                        val_pairs = val_pairs_sg
                    else:
                        train_pairs = train_pairs_cbow
                        val_pairs = val_pairs_cbow
                    out = train_model(
                        model_type=model_type,
                        vocab_size=vocab_size,
                        embedding_dim=d,
                        train_pairs=train_pairs,
                        val_pairs=val_pairs,
                        lr=lr,
                        epochs=epochs,
                        batch_size=bs,
                        seed=seed,
                    )
                    final_val = out["history"]["val_loss"][-1]
                    result = {
                        "model_type": model_type,
                        "embedding_dim": d,
                        "lr": lr,
                        "batch_size": bs,
                        "final_val_loss": final_val,
                        "history": out["history"],
                    }
                    logging.info(f"Result: {result}")
                    results.append(result)
    return results

