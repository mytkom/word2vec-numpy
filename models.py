from typing import Tuple

import numpy as np


class Word2VecBase:
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 123):
        rng = np.random.RandomState(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.W_in = 0.01 * rng.randn(vocab_size, embedding_dim).astype(np.float32)
        self.W_out = 0.01 * rng.randn(embedding_dim, vocab_size).astype(np.float32)

    def parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.W_in, self.W_out


class SkipGram(Word2VecBase):
    def forward(self, centers: np.ndarray, contexts: np.ndarray) -> Tuple[float, Tuple]:
        h = self.W_in[centers]  # (batch, D)
        logits = h @ self.W_out  # (batch, V)
        # numerical stability
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        batch_size = centers.shape[0]
        loss = -np.log(probs[np.arange(batch_size), contexts] + 1e-10).mean()
        cache = (h, probs, centers, contexts)
        return float(loss), cache

    def backward(self, cache: Tuple, lr: float):
        h, probs, centers, contexts = cache
        batch_size = centers.shape[0]
        grad_logits = probs
        grad_logits[np.arange(batch_size), contexts] -= 1.0
        grad_logits /= batch_size

        grad_W_out = h.T @ grad_logits  # (D, V)
        grad_h = grad_logits @ self.W_out.T  # (batch, D)

        grad_W_in = np.zeros_like(self.W_in)
        np.add.at(grad_W_in, centers, grad_h)

        self.W_in -= lr * grad_W_in
        self.W_out -= lr * grad_W_out


class CBOW(Word2VecBase):
    def forward(self, contexts: np.ndarray, targets: np.ndarray) -> Tuple[float, Tuple]:
        # contexts: (batch, context_len)
        h = self.W_in[contexts]  # (batch, context_len, D)
        h_mean = h.mean(axis=1)  # (batch, D)
        logits = h_mean @ self.W_out  # (batch, V)
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        batch_size = targets.shape[0]
        loss = -np.log(probs[np.arange(batch_size), targets] + 1e-10).mean()
        cache = (h, h_mean, probs, contexts, targets)
        return float(loss), cache

    def backward(self, cache: Tuple, lr: float):
        h, h_mean, probs, contexts, targets = cache
        batch_size, context_len, _ = h.shape
        grad_logits = probs
        grad_logits[np.arange(batch_size), targets] -= 1.0
        grad_logits /= batch_size

        grad_W_out = h_mean.T @ grad_logits  # (D, V)
        grad_h_mean = grad_logits @ self.W_out.T  # (batch, D)
        grad_h = (grad_h_mean[:, None, :] / context_len).repeat(context_len, axis=1)

        grad_W_in = np.zeros_like(self.W_in)
        for i in range(batch_size):
            np.add.at(grad_W_in, contexts[i], grad_h[i])

        self.W_in -= lr * grad_W_in
        self.W_out -= lr * grad_W_out

