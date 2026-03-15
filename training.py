from typing import List, Dict, Any, Callable, Type
import logging
from dataclasses import dataclass
import time

import numpy as np

from models import (
    SkipGram,
    CBOW,
    SkipGramNS,
    CBOWNS,
    Word2VecBase,
    Word2VecBaseNS,
)
from pair_providers import BasePairProvider, SkipGramPairProvider, CBOWPairProvider


@dataclass
class ModelSpec:
    name: str
    model_cls: Type[Word2VecBase]
    pair_provider_cls: Type[BasePairProvider]
    batcher: Callable
    num_neg: int = 5


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "skipgram": ModelSpec(
        name="skipgram",
        model_cls=SkipGram,
        pair_provider_cls=SkipGramPairProvider,
        batcher=SkipGramPairProvider.batch,
    ),
    "cbow": ModelSpec(
        name="cbow",
        model_cls=CBOW,
        pair_provider_cls=CBOWPairProvider,
        batcher=CBOWPairProvider.batch,
    ),
    "skipgram_ns": ModelSpec(
        name="skipgram_ns",
        model_cls=SkipGramNS,
        pair_provider_cls=SkipGramPairProvider,
        batcher=SkipGramPairProvider.batch,
        num_neg=5,
    ),
    "cbow_ns": ModelSpec(
        name="cbow_ns",
        model_cls=CBOWNS,
        pair_provider_cls=CBOWPairProvider,
        batcher=CBOWPairProvider.batch,
        num_neg=5,
    ),
}


def train_model(
    spec: ModelSpec,
    vocab_size: int,
    embedding_dim: int,
    train_pairs,
    val_pairs,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int = 123,
    neg_probs: np.ndarray | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    lr_decay_final_factor: float = 1.0,
) -> Dict[str, Any]:
    model = spec.model_cls(vocab_size, embedding_dim, seed=seed)
    batcher = spec.batcher
    use_ns = issubclass(spec.model_cls, Word2VecBaseNS)
    rng = np.random.RandomState(seed)

    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    best_val = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        if epochs > 1 and lr_decay_final_factor != 1.0:
            frac = epoch / (epochs - 1)
            lr_factor = 1.0 + (lr_decay_final_factor - 1.0) * frac
        else:
            lr_factor = 1.0
        lr_epoch = lr * lr_factor
        epoch_start = time.perf_counter()
        train_losses = []
        for xb, yb in batcher(train_pairs, batch_size=batch_size, seed=seed + epoch):
            if use_ns:
                if neg_probs is None:
                    raise ValueError("neg_probs must be provided for negative sampling models")
                neg_targets = rng.choice(
                    vocab_size,
                    size=(xb.shape[0], spec.num_neg),
                    p=neg_probs,
                ).astype(np.int64)
                loss, cache = model.forward(xb, yb, neg_targets)
            else:
                loss, cache = model.forward(xb, yb)
            model.backward(cache, lr_epoch)
            train_losses.append(loss)
        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")

        val_losses = []
        for xb, yb in batcher(val_pairs, batch_size=batch_size, seed=seed + 999 + epoch):
            if use_ns:
                if neg_probs is None:
                    raise ValueError("neg_probs must be provided for negative sampling models")
                neg_targets = rng.choice(
                    vocab_size,
                    size=(xb.shape[0], spec.num_neg),
                    p=neg_probs,
                ).astype(np.int64)
                loss, _ = model.forward(xb, yb, neg_targets)
            else:
                loss, _ = model.forward(xb, yb)
            val_losses.append(loss)
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")

        epoch_time = time.perf_counter() - epoch_start

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)
        history["epoch_time"].append(epoch_time)

        logging.info(f"Epoch {epoch} time: {epoch_time:.2f}s, train loss: {mean_train:.4f}, val loss: {mean_val:.4f}")

        if mean_val < best_val - early_stopping_min_delta:
            best_val = mean_val
            no_improve = 0
        else:
            no_improve += 1

        if early_stopping_patience is not None and no_improve >= early_stopping_patience:
            break

    return {"model": model, "history": history}


def grid_search_for_model(
    spec: ModelSpec,
    vocab_size: int,
    train_pairs,
    val_pairs,
    model_cfg,
    seed: int = 123,
    neg_probs: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    def _to_list(x):
        if x is None:
            return []
        return x if isinstance(x, list) else [x]

    embedding_dims = _to_list(model_cfg.embedding_dims)
    lrs = _to_list(model_cfg.lrs)
    batch_sizes = _to_list(model_cfg.batch_sizes)
    lr_decay_final_factors = _to_list(model_cfg.lr_decay_final_factor)

    # handle num_neg grid only for NS models
    is_ns = issubclass(spec.model_cls, Word2VecBaseNS)
    if is_ns:
        if model_cfg.num_neg is not None:
            num_negs = _to_list(model_cfg.num_neg)
        else:
            num_negs = [spec.num_neg]
    else:
        num_negs = [spec.num_neg]

    results: List[Dict[str, Any]] = []
    for d in embedding_dims:
        for lr in lrs:
            for bs in batch_sizes:
                for nneg in num_negs:
                    for lr_decay_final_factor in lr_decay_final_factors:
                        local_spec = spec
                        if is_ns and nneg != spec.num_neg:
                            local_spec = ModelSpec(
                                name=spec.name,
                                model_cls=spec.model_cls,
                                pair_provider_cls=spec.pair_provider_cls,
                                batcher=spec.batcher,
                                num_neg=nneg,
                            )
                        logging.info(
                            f"Training {local_spec.name} with dim={d}, lr={lr}, bs={bs}"
                            + (f", num_neg={nneg}" if is_ns else "")
                        )
                        out = train_model(
                            spec=local_spec,
                            vocab_size=vocab_size,
                            embedding_dim=d,
                            train_pairs=train_pairs,
                            val_pairs=val_pairs,
                            lr=lr,
                            epochs=model_cfg.epochs,
                            batch_size=bs,
                            seed=seed,
                            neg_probs=neg_probs,
                            early_stopping_patience=model_cfg.early_stopping_patience,
                            early_stopping_min_delta=model_cfg.early_stopping_min_delta,
                            lr_decay_final_factor=lr_decay_final_factor,
                        )
                        final_val = out["history"]["val_loss"][-1]
                        epoch_times = out["history"].get("epoch_time", [])
                        mean_epoch_time = float(np.mean(epoch_times)) if epoch_times else float("nan")
                        result = {
                            "model_type": local_spec.name,
                            "embedding_dim": d,
                            "lr": lr,
                            "batch_size": bs,
                            "final_val_loss": final_val,
                            "history": out["history"],
                            "model": out["model"],
                            "mean_epoch_time": mean_epoch_time,
                        }
                        if is_ns:
                            result["num_neg"] = nneg
                        logging.info(f"Result: {result}")
                        results.append(result)
    return results

