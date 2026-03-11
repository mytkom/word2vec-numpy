import numpy as np
import os
import logging

from data_utils import (
    fetch_shakespeare,
    tokenize,
    build_vocab,
    encode_tokens,
    generate_skipgram_pairs,
    generate_cbow_pairs,
    train_val_split,
)
from training import grid_search

logging.basicConfig(level=logging.DEBUG)

def main():
    logging.info("Fetching Shakespeare corpus and building vocabulary")
    raw = fetch_shakespeare()

    logging.debug("Tokenizing Shakespeare corpus")
    tokens = tokenize(raw)

    logging.debug("Building vocabulary")
    word_to_id, id_to_word = build_vocab(tokens, max_vocab_size=4096)

    logging.debug("Encoding tokens")
    ids = encode_tokens(tokens, word_to_id)
    
    logging.debug("Generating skipgram and CBOW pairs")

    # use a subset for speed
    max_tokens = 50000
    ids = ids[:max_tokens]

    window_size = 2

    sg_pairs = generate_skipgram_pairs(ids, window_size=window_size)
    cbow_pairs = generate_cbow_pairs(ids, window_size=window_size)

    sg_indices = np.arange(len(sg_pairs))
    cbow_indices = np.arange(len(cbow_pairs))

    logging.debug("Splitting skipgram and CBOW pairs into train and validation sets")

    sg_train_idx, sg_val_idx = train_val_split(sg_indices)
    cbow_train_idx, cbow_val_idx = train_val_split(cbow_indices)

    sg_train_pairs = [sg_pairs[i] for i in sg_train_idx]
    sg_val_pairs = [sg_pairs[i] for i in sg_val_idx]
    cbow_train_pairs = [cbow_pairs[i] for i in cbow_train_idx]
    cbow_val_pairs = [cbow_pairs[i] for i in cbow_val_idx]

    embedding_dims = [32, 64]
    lrs = [0.05, 0.1]
    batch_sizes = [64, 128]
    epochs = 3

    logging.info("Running grid search")

    results = grid_search(
        vocab_size=len(word_to_id),
        train_pairs_sg=sg_train_pairs,
        val_pairs_sg=sg_val_pairs,
        train_pairs_cbow=cbow_train_pairs,
        val_pairs_cbow=cbow_val_pairs,
        embedding_dims=embedding_dims,
        lrs=lrs,
        batch_sizes=batch_sizes,
        epochs=epochs,
    )

    logging.info("Sorting results by validation loss")

    results_sorted = sorted(results, key=lambda r: r["final_val_loss"])
    for r in results_sorted:
        print(
            f"{r['model_type']}, dim={r['embedding_dim']}, lr={r['lr']}, "
            f"bs={r['batch_size']}, val_loss={r['final_val_loss']:.4f}"
        )


if __name__ == "__main__":
    main()

