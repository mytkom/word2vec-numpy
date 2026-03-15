from typing import Tuple
from abc import ABC, abstractmethod

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


class Word2VecBaseSoftmax(Word2VecBase, ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray, targets: np.ndarray) -> Tuple[float, Tuple]:
        raise NotImplementedError

    @abstractmethod
    def backward(self, cache: Tuple, lr: float):
        raise NotImplementedError


class Word2VecBaseNS(Word2VecBase, ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray, pos_targets: np.ndarray, neg_targets: np.ndarray) -> Tuple[float, Tuple]:
        raise NotImplementedError

    @abstractmethod
    def backward(self, cache: Tuple, lr: float):
        raise NotImplementedError


class SkipGram(Word2VecBaseSoftmax):
    """Skip-gram model"""
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

        self.W_in -= lr * batch_size * grad_W_in
        self.W_out -= lr * batch_size * grad_W_out


class CBOW(Word2VecBaseSoftmax):
    """Continuous Bag of Words model"""
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

        self.W_in -= lr * batch_size * grad_W_in
        self.W_out -= lr * batch_size * grad_W_out


class SkipGramNS(Word2VecBaseNS):
    """Skip-gram with negative sampling"""
    def forward(
        self,
        centers: np.ndarray,
        pos_contexts: np.ndarray,
        neg_contexts: np.ndarray,
    ) -> Tuple[float, Tuple]:
        h = self.W_in[centers]  # (batch, D)

        pos_w = self.W_out[:, pos_contexts]  # (D, batch)
        pos_score = np.sum(h * pos_w.T, axis=1)  # (batch,)
        pos_logits = pos_score

        neg_w = self.W_out[:, neg_contexts]  # (D, batch, K)
        neg_score = np.einsum("bd,dbk->bk", h, neg_w)  # (batch, K)
        neg_logits = -neg_score

        pos_loss = -np.log(1.0 / (1.0 + np.exp(-pos_logits)) + 1e-10)
        neg_loss = -np.log(1.0 - 1.0 / (1.0 + np.exp(-neg_logits)) + 1e-10)
        loss = (pos_loss.mean() + neg_loss.mean())

        cache = (h, centers, pos_contexts, neg_contexts, pos_logits, neg_logits)
        return float(loss), cache

    def backward(self, cache: Tuple, lr: float):
        h, centers, pos_contexts, neg_contexts, pos_logits, neg_logits = cache
        batch_size = centers.shape[0]
        num_neg = neg_contexts.shape[1]

        pos_sig = 1.0 / (1.0 + np.exp(-pos_logits))
        neg_sig = 1.0 / (1.0 + np.exp(-neg_logits))

        dpos = (pos_sig - 1.0) / batch_size
        dneg = neg_sig / (batch_size * num_neg)

        pos_w = self.W_out[:, pos_contexts]  # (D, batch)
        grad_W_out_pos = np.zeros_like(self.W_out)
        for i in range(batch_size):
            idx = pos_contexts[i]
            grad_W_out_pos[:, idx] += dpos[i] * h[i]

        grad_h_pos = dpos[:, None] * pos_w.T

        neg_w = self.W_out[:, neg_contexts]  # (D, batch, K)
        grad_W_out_neg = np.zeros_like(self.W_out)
        for i in range(batch_size):
            for k in range(num_neg):
                idx = neg_contexts[i, k]
                grad_W_out_neg[:, idx] += (-dneg[i, k]) * h[i]

        grad_h_neg = np.zeros_like(h)
        for i in range(batch_size):
            grad_h_neg[i] += np.sum((-dneg[i, :, None]) * neg_w[:, i, :].T, axis=0)

        grad_h = grad_h_pos + grad_h_neg

        grad_W_in = np.zeros_like(self.W_in)
        np.add.at(grad_W_in, centers, grad_h)

        self.W_in -= lr * batch_size * grad_W_in
        self.W_out -= lr * batch_size * (grad_W_out_pos + grad_W_out_neg)


class CBOWNS(Word2VecBaseNS):
    """CBOW with negative sampling"""
    def forward(
        self,
        contexts: np.ndarray,
        pos_targets: np.ndarray,
        neg_targets: np.ndarray,
    ) -> Tuple[float, Tuple]:
        h = self.W_in[contexts]  # (batch, context_len, D)
        h_mean = h.mean(axis=1)  # (batch, D)

        pos_w = self.W_out[:, pos_targets]  # (D, batch)
        pos_score = np.sum(h_mean * pos_w.T, axis=1)  # (batch,)
        pos_logits = pos_score

        neg_w = self.W_out[:, neg_targets]  # (D, batch, K)
        neg_score = np.einsum("bd,dbk->bk", h_mean, neg_w)  # (batch, K)
        neg_logits = -neg_score

        pos_loss = -np.log(1.0 / (1.0 + np.exp(-pos_logits)) + 1e-10)
        neg_loss = -np.log(1.0 - 1.0 / (1.0 + np.exp(-neg_logits)) + 1e-10)
        loss = (pos_loss.mean() + neg_loss.mean())

        cache = (h, h_mean, contexts, pos_targets, neg_targets, pos_logits, neg_logits)
        return float(loss), cache

    def backward(self, cache: Tuple, lr: float):
        h, h_mean, contexts, pos_targets, neg_targets, pos_logits, neg_logits = cache
        batch_size, context_len, _ = h.shape
        num_neg = neg_targets.shape[1]

        pos_sig = 1.0 / (1.0 + np.exp(-pos_logits))
        neg_sig = 1.0 / (1.0 + np.exp(-neg_logits))

        dpos = (pos_sig - 1.0) / batch_size
        dneg = neg_sig / (batch_size * num_neg)

        pos_w = self.W_out[:, pos_targets]
        grad_W_out_pos = np.zeros_like(self.W_out)
        for i in range(batch_size):
            idx = pos_targets[i]
            grad_W_out_pos[:, idx] += dpos[i] * h_mean[i]

        grad_h_mean_pos = dpos[:, None] * pos_w.T

        neg_w = self.W_out[:, neg_targets]
        grad_W_out_neg = np.zeros_like(self.W_out)
        grad_h_mean_neg = np.zeros_like(h_mean)
        for i in range(batch_size):
            for k in range(num_neg):
                idx = neg_targets[i, k]
                grad_W_out_neg[:, idx] += (-dneg[i, k]) * h_mean[i]
                grad_h_mean_neg[i] += (-dneg[i, k]) * neg_w[:, i, k]

        grad_h_mean = grad_h_mean_pos + grad_h_mean_neg
        grad_h = (grad_h_mean[:, None, :] / context_len).repeat(context_len, axis=1)

        grad_W_in = np.zeros_like(self.W_in)
        for i in range(batch_size):
            np.add.at(grad_W_in, contexts[i], grad_h[i])

        self.W_in -= lr * batch_size * grad_W_in
        self.W_out -= lr * batch_size * (grad_W_out_pos + grad_W_out_neg)

