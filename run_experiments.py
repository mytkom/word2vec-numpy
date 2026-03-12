
import logging
import json
import csv
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import tyro
from dataclass_wizard import JSONWizard

from data_utils import (
    fetch_shakespeare,
    tokenize,
    build_vocab,
    encode_tokens,
)
from training import MODEL_REGISTRY, grid_search_for_model


logging.basicConfig(level=logging.DEBUG)


@dataclass
class ModelGridConfig:
    name: str
    embedding_dims: List[int] = field(default_factory=lambda: [32, 64])
    lrs: List[float] = field(default_factory=lambda: [0.05, 0.1])
    batch_sizes: List[int] = field(default_factory=lambda: [64, 128])
    epochs: int = 3


@dataclass
class DataConfig:
    max_vocab_size: int = 4096
    max_tokens: int = 50000
    window_size: int = 2
    val_ratio: float = 0.1
    seed: int = 123


@dataclass
class ExperimentConfig(JSONWizard):
    data: DataConfig = field(default_factory=DataConfig)
    models: List[ModelGridConfig] = field(
        default_factory=lambda: [
            ModelGridConfig(name="skipgram"),
            ModelGridConfig(name="cbow"),
        ]
    )
    config_json: Optional[str] = None


def load_or_build_dataset(data_cfg: DataConfig) -> (Dict[str, int], np.ndarray):
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    vocab_path = cache_dir / f"word_to_id_{data_cfg.max_vocab_size}.json"
    ids_path = cache_dir / f"ids_{data_cfg.max_vocab_size}_{data_cfg.max_tokens}.npy"

    if vocab_path.exists() and ids_path.exists():
        logging.info("Loading cached dataset")
        with open(vocab_path, "r") as f:
            word_to_id: Dict[str, int] = json.load(f)
        ids = np.load(ids_path)
        return word_to_id, ids

    logging.info("Fetching Shakespeare corpus and building vocabulary")
    raw = fetch_shakespeare()

    logging.debug("Tokenizing Shakespeare corpus")
    tokens = tokenize(raw)

    logging.debug("Building vocabulary")
    word_to_id, id_to_word = build_vocab(tokens, max_vocab_size=data_cfg.max_vocab_size)

    logging.debug("Encoding tokens")
    ids = encode_tokens(tokens, word_to_id)

    max_tokens = data_cfg.max_tokens
    if max_tokens is not None:
        ids = ids[:max_tokens]

    with open(vocab_path, "w") as f:
        json.dump(word_to_id, f)
    np.save(ids_path, ids)

    return word_to_id, ids


def run_experiment(cfg: ExperimentConfig):
    word_to_id, ids = load_or_build_dataset(cfg.data)

    results_all = []

    for model_cfg in cfg.models:
        key = model_cfg.name.lower()
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model name {model_cfg.name}")
        spec = MODEL_REGISTRY[key]

        logging.debug(f"Preparing pair provider for model {model_cfg.name}")
        provider = spec.pair_provider_cls(
            ids=ids,
            window_size=cfg.data.window_size,
            val_ratio=cfg.data.val_ratio,
            seed=cfg.data.seed,
        )
        train_pairs, val_pairs = provider.get_train_val_pairs()

        logging.info(f"Running grid search for model {model_cfg.name}")
        results = grid_search_for_model(
            spec=spec,
            vocab_size=len(word_to_id),
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            embedding_dims=model_cfg.embedding_dims,
            lrs=model_cfg.lrs,
            batch_sizes=model_cfg.batch_sizes,
            epochs=model_cfg.epochs,
            seed=cfg.data.seed,
        )
        results_all.extend(results)

    logging.info("Sorting results by validation loss")
    results_sorted = sorted(results_all, key=lambda r: r["final_val_loss"])

    for r in results_sorted:
        print(
            f"{r['model_type']}, dim={r['embedding_dim']}, lr={r['lr']}, "
            f"bs={r['batch_size']}, val_loss={r['final_val_loss']:.4f}"
        )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "results.csv"
    logging.info(f"Saving all results to {csv_path}")
    fieldnames = ["model_type", "embedding_dim", "lr", "batch_size", "final_val_loss", "train_loss", "val_loss"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_sorted:
            history = r["history"]
            row = {
                "model_type": r["model_type"],
                "embedding_dim": r["embedding_dim"],
                "lr": r["lr"],
                "batch_size": r["batch_size"],
                "final_val_loss": r["final_val_loss"],
                "train_loss": json.dumps(history["train_loss"]),
                "val_loss": json.dumps(history["val_loss"]),
            }
            writer.writerow(row)

    best_models: Dict[str, Dict[str, Any]] = {}
    for r in results_sorted:
        mtype = r["model_type"]
        if mtype not in best_models or r["final_val_loss"] < best_models[mtype]["final_val_loss"]:
            best_models[mtype] = {
                "final_val_loss": r["final_val_loss"],
                "config": {
                    "embedding_dim": r["embedding_dim"],
                    "lr": r["lr"],
                    "batch_size": r["batch_size"],
                },
                "history": r["history"],
                "model": r["model"],
            }

    pickle_path = output_dir / "best_models.pkl"
    logging.info(f"Saving best models to {pickle_path}")
    with open(pickle_path, "wb") as f:
        pickle.dump(best_models, f)


def main(cfg: ExperimentConfig):
    if cfg.config_json:
        logging.info(f"Loading configuration from {cfg.config_json}")
        with open(cfg.config_json, "r") as f:
            data = json.load(f)
        cfg = ExperimentConfig.from_dict(data)
    run_experiment(cfg)


if __name__ == "__main__":
    tyro.cli(main)

