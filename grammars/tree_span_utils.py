import numpy as np
import random
from pcfg_base import SimpleTree

catalans = np.ones(100)
for i in range(1, 100):
    catalans[i] = (catalans[:i] * catalans[i - 1 :: -1]).sum()


def uniform_random_imgtree(n):
    grid = np.zeros((n, n))
    grid[0, -1] = 2
    if n > 1:
        p = catalans[: n - 1] * catalans[n - 2 :: -1]
        pos = np.random.choice(np.arange(n - 1), p=p / p.sum())
        grid[: pos + 1, : pos + 1] = uniform_random_imgtree(pos + 1)
        grid[pos + 1 :, pos + 1 :] = uniform_random_imgtree(n - pos - 1)
    return grid


def spans_to_imgtree(tree):
    l = max([s[1] for s in tree]) + 1
    grid = np.eye(l) * 2
    for a, b in tree:
        grid[a, b] = 2
    return grid


def binarize_imgtree(grid_in):
    grid = np.copy(grid_in)
    parentless = []
    for a, b in zip(*np.nonzero(grid)):
        up = grid[:a, b] == 2
        right = grid[a, b + 1 :] == 2
        assert not (np.any(up) and np.any(right))
        if np.any(up):
            grid[np.nonzero(up)[0][-1] + 1 : a, b] = 1
        elif np.any(right):
            grid[a, b + 1 : np.nonzero(right)[0][0] + b + 1] = 1
        elif a != 0:
            parentless.append((a, b))
    while len(parentless) > 0:
        a, b = random.choice(parentless)
        up = grid[:a, b] == 1
        right = grid[a, b + 1 :] == 1
        assert np.any(up) and np.any(right)
        if np.random.randint(0, 1) == 0:
            grid[np.nonzero(up)[0][-1] + 1 : a, b] = 1
            grid[np.nonzero(up)[0][-1], b] = 2
        else:
            grid[a, b + 1 : np.nonzero(right)[0][0] + b + 1] = 1
            grid[a, np.nonzero(right)[0][0] + b + 1] = 2
        parentless.remove((a, b))
    return grid


def imgtree_to_tree(grid):
    if grid.shape == (1, 1):
        return SimpleTree(0)

    l = np.flatnonzero(grid[0, :-1] == 2).max()
    left = imgtree_to_tree(grid[: l + 1, : l + 1])

    assert np.flatnonzero(grid[1:, -1] == 2).min() == l
    right = imgtree_to_tree(grid[l + 1 :, l + 1 :])

    return SimpleTree(0, left, right)


def tree_to_spans(tree):
    return [(span[0], span[0] + span[1] - 1) for span in tree.all_spans(0)[0]]
