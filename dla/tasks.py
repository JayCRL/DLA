"""Self-contained continual-learning task families.

Two synthetic families are provided so the experiments never depend on an external
dataset download:

* ``gaussian`` - two class-conditional Gaussian clouds (linearly separable in
  expectation, but the separating direction is random per task).
* ``xor``      - four Gaussian clusters arranged in XOR configuration (needs the
  hidden layer of the Neural Core, not solvable by a linear rule).

Each task is a small "episode of experience": a short train stream plus a held-out
test set.  Continual learning means the model sees one task after another and is
evaluated on all previous tasks (forgetting / backward transfer).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Task:
    name: str
    family: str
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor

    @property
    def n_train(self) -> int:
        return self.x_train.shape[0]

    @property
    def n_test(self) -> int:
        return self.x_test.shape[0]

    def train_batches(self, batch_size: int, seed: int = 0, shuffle: bool = True):
        """Deterministic mini-batches over the training experience."""
        n = self.n_train
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)) if shuffle else torch.arange(n)
        for i in range(0, n, batch_size):
            j = idx[i : i + batch_size]
            yield self.x_train[j], self.y_train[j]

    def to(self, device):
        self.x_train = self.x_train.to(device)
        self.y_train = self.y_train.to(device)
        self.x_test = self.x_test.to(device)
        self.y_test = self.y_test.to(device)
        return self


def make_gaussian_task(input_dim: int, seed: int, n_train: int = 80, n_test: int = 200, sep: float = 1.0, noise: float = 0.9) -> Task:
    """Two Gaussian clouds along a random direction."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(input_dim, generator=g)
    u = u / (u.norm() + 1e-8)

    def sample(n: int):
        half = n // 2
        y = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(n - half, dtype=torch.long)])
        means = torch.where(y.unsqueeze(1) == 0, -sep * u, sep * u)
        x = means + noise * torch.randn(n, input_dim, generator=g)
        perm = torch.randperm(n, generator=g)
        return x[perm], y[perm]

    x_tr, y_tr = sample(n_train)
    x_te, y_te = sample(n_test)
    return Task(f"gauss-{seed}", "gaussian", x_tr, y_tr, x_te, y_te)


def make_xor_task(input_dim: int, seed: int, n_train: int = 80, n_test: int = 200, scale: float = 1.3, noise: float = 0.45) -> Task:
    """XOR of two quadrant signs on the first two dimensions + noise dims."""
    g = torch.Generator().manual_seed(seed)

    def sample(n: int):
        q1 = 2 * torch.randint(0, 2, (n,), generator=g) - 1  # -1 / +1
        q2 = 2 * torch.randint(0, 2, (n,), generator=g) - 1
        y = (q1 != q2).long()
        x = noise * torch.randn(n, input_dim, generator=g)
        x[:, 0] = scale * q1.float() + noise * torch.randn(n, generator=g)
        x[:, 1] = scale * q2.float() + noise * torch.randn(n, generator=g)
        return x, y

    x_tr, y_tr = sample(n_train)
    x_te, y_te = sample(n_test)
    return Task(f"xor-{seed}", "xor", x_tr, y_tr, x_te, y_te)


def make_task_stream(
    n_meta_train: int,
    n_eval: int,
    input_dim: int = 6,
    seed_base: int = 1234,
    families=("gaussian", "xor"),
    n_train: int = 80,
    n_test: int = 200,
):
    """Returns (meta_train_tasks, eval_tasks) with disjoint task seeds."""
    factories = {"gaussian": make_gaussian_task, "xor": make_xor_task}
    meta_train, eval_tasks = [], []
    for i in range(n_meta_train):
        family = families[i % len(families)]
        meta_train.append(factories[family](input_dim, seed_base + 1000 * (i + 1), n_train, n_test))
    for i in range(n_eval):
        family = families[i % len(families)]
        eval_tasks.append(factories[family](input_dim, seed_base + 1000 * (n_meta_train + i + 1), n_train, n_test))
    return meta_train, eval_tasks
