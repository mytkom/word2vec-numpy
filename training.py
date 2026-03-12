from typing import List, Dict, Any, Callable, Type
import logging
from dataclasses import dataclass

import numpy as np

from models import SkipGram, CBOW, Word2VecBase
from pair_providers import BasePairProvider, SkipGramPairProvider, CBOWPairProvider


@dataclass
class ModelSpec:
    name: str
    model_cls: Type[Word2VecBase]
    pair_provider_cls: Type[BasePairProvider]
    batcher: Callable


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
) -> Dict[str, Any]:
    model = spec.model_cls(vocab_size, embedding_dim, seed=seed)
    batcher = spec.batcher

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        train_losses = []
        for xb, yb in batcher(train_pairs, batch_size=batch_size, seed=seed + epoch):
            loss, cache = model.forward(xb, yb)
            model.backward(cache, lr)
            train_losses.append(loss)
        mean_train = float(np.mean(train_losses)) if train_losses else float("nan")

        val_losses = []
        for xb, yb in batcher(val_pairs, batch_size=batch_size, seed=seed + 999 + epoch):
            loss, _ = model.forward(xb, yb)
            val_losses.append(loss)
        mean_val = float(np.mean(val_losses)) if val_losses else float("nan")

        history["train_loss"].append(mean_train)
        history["val_loss"].append(mean_val)

    return {"model": model, "history": history}


def grid_search_for_model(
    spec: ModelSpec,
    vocab_size: int,
    train_pairs,
    val_pairs,
    embedding_dims,
    lrs,
    batch_sizes,
    epochs: int,
    seed: int = 123,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for d in embedding_dims:
        for lr in lrs:
            for bs in batch_sizes:
                logging.info(f"Training {spec.name} with dim={d}, lr={lr}, bs={bs}")
                out = train_model(
                    spec=spec,
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
                    "model_type": spec.name,
                    "embedding_dim": d,
                    "lr": lr,
                    "batch_size": bs,
                    "final_val_loss": final_val,
                    "history": out["history"],
                    "model": out["model"],
                }
                logging.info(f"Result: {result}")
                results.append(result)
    return results

