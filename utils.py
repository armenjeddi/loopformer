import os, numpy as np, torch, random
from torch.utils.data import Dataset, DataLoader, Sampler

class MemmapTokenDataset(Dataset):
    """
    Produces (x,y) token windows from a uint16 memmap in a deterministic order.
    Order is controlled by `starts`, which you pass in (sequential or permuted).
    """
    def __init__(self, path: str, block_size: int, starts: np.ndarray):
        self.path = path
        self.block_size = block_size
        self.starts = starts.astype(np.int64)
        self._mmap = None  # lazily opened per worker

    def _ensure_open(self):
        if self._mmap is None:
            self._mmap = np.memmap(self.path, dtype=np.uint16, mode='r')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        self._ensure_open()
        i = int(self.starts[idx])
        x = torch.from_numpy(self._mmap[i:i+self.block_size].astype(np.int64))
        y = torch.from_numpy(self._mmap[i+1:i+1+self.block_size].astype(np.int64))
        return x, y

def make_starts(n_tokens: int, block_size: int, stride: int = None):
    # sequential, deterministic; no overlap if stride==block_size
    if stride is None:
        stride = block_size
    max_start = n_tokens - block_size - 1
    return np.arange(0, max_start + 1, stride, dtype=np.int64)

def permute_starts(starts: np.ndarray, seed: int):
    g = np.random.RandomState(seed)
    perm = starts.copy()
    g.shuffle(perm)
    return perm

class FixedOrderSampler(Sampler[int]):
    def __init__(self, order: np.ndarray):
        self.order = order.tolist()
    def __iter__(self):
        return iter(self.order)
    def __len__(self):
        return len(self.order)


def count_tokens(bin_path: str) -> int:
    # uint16 tokens
    return os.path.getsize(bin_path) // np.dtype(np.uint16).itemsize

def build_loader(data_dir, split, block_size, batch_size,
                 mode='sequential', stride=None, seed=1337,
                 num_workers=2, persistent_workers=True, drop_last=True):
    path = os.path.join(data_dir, f'{split}.bin')
    n_tokens = count_tokens(path)
    print(f"n_tokens {split}: {n_tokens}")
    starts = make_starts(n_tokens, block_size, stride=stride)

    if mode == 'permuted':
        starts = permute_starts(starts, seed)

    ds = MemmapTokenDataset(path, block_size, starts)

    if mode == 'permuted':
        sampler = FixedOrderSampler(np.arange(len(ds)))  # order already baked into starts
        shuffle = False
    else:
        sampler = None
        shuffle = False  # strict sequential order

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )
    return loader

def seed_everything(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_short_trajectory(max_loops):
    """
    Return a random discrete trajectory from 0 to 1, as an array of step sizes
    drawn from {x / max_loops | x=1..max_loops-1}, with total sum == 1.

    The number of steps (trajectory length) L is chosen uniformly from 1 to max_loops-1.
    """
    assert max_loops >= 2, "max_loops must be >= 2"

    allowed = [x / max_loops for x in range(1, max_loops+1)]  # possible step sizes
    min_step = allowed[0]

    # choose path length L uniformly
    L = random.randint(1, max_loops - 1)

    trajectory = []
    remaining = 1.0
    remaining_steps = L

    for _ in range(L - 1):
        # max we can take now so that (remaining_steps - 1) * min_step can still fit
        cap = remaining - (remaining_steps - 1) * min_step
        # pick any allowed step <= cap
        choices = [s for s in allowed if s <= cap + 1e-12]
        v = random.choice(choices)
        trajectory.append(v)
        remaining -= v
        remaining_steps -= 1

    # last step must exactly fill to 1
    trajectory.append(remaining)

    return np.array(trajectory)

