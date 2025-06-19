import random
import typing as tp

import numpy as np


def identity(board: list[list[int]]) -> np.ndarray:
    return np.array(board)


def hflip(board: list[list[int]]) -> np.ndarray:
    return np.fliplr(board)


def vflip(board: list[list[int]]) -> np.ndarray:
    return np.flipud(board)


def dflip(board: list[list[int]]) -> np.ndarray:
    return np.array([row[::-1] for row in board[::-1]])


def transpose(board: list[list[int]]) -> np.ndarray:
    return np.array(board).T


def rot90(board: list[list[int]]) -> np.ndarray:
    return np.rot90(board)


def rot270(board: list[list[int]]) -> np.ndarray:
    return np.rot90(board, k=3)


def color_remap(board: list[list[int]], color_permutation: dict | None = None) -> np.ndarray:
    if color_permutation is None:
        permute_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # keep black intact
        random_color_permutation = np.arange(10)
        random_color_permutation[permute_colors] = np.random.permutation(random_color_permutation[permute_colors])

        color_permutation = {i: color for i, color in enumerate(random_color_permutation)}

    # https://stackoverflow.com/a/16992783
    return np.vectorize(color_permutation.get)(board)


# def repeat(board, times: tuple[int, int] | None = None):
#     if times is None:
#         times = random.choice([(1, 2), (2, 1), (2, 2)])
#     return np.tile(board, times)


# def inflate(board, scale: tuple[int, int] | None = None):
#     if scale is None:
#         scale = random.choice([(1, 2), (2, 1), (2, 2)])
#     # https://stackoverflow.com/a/7656683
#     return np.kron(board, np.ones(scale, dtype=int))


# def grid(board, every=(1, 1), width=1, color: int | None = None):
#     if color is None:
#         color = random.choice(range(10))

#     shape = np.shape(board)
#     assert shape[0] % every[0] == 0
#     assert shape[1] % every[1] == 0

#     gboard = []
#     _hotizontal_grid_block = [[color] * (shape[1] + width * (shape[1] // every[1] + 1)) for _ in range(width)]
#     for i in range(shape[0]):
#         if i % every[0] == 0:
#             gboard.extend(_hotizontal_grid_block)
#         grow = []
#         for j in range(shape[1]):
#             if j % every[1] == 0:
#                 grow.extend([color] * width)
#             grow.append(board[i][j])
#         grow.extend([color] * width)
#         gboard.append(grow)
#     gboard.extend(_hotizontal_grid_block)
#     return gboard


all_augmentations = [identity, hflip, vflip, dflip, transpose, rot90, rot270, color_remap]


def choose_random_augmentation():
    random_aumgentation_fn = np.random.choice(all_augmentations)
    return random_aumgentation_fn


def apply_augmentation_fn(sample, augmentation_fn):
    new_sample = {'train': [], 'test': []}
    for train_sample in sample['train']:
        new_sample['train'].append(
            {
                'input': augmentation_fn(train_sample['input']),
                'output': augmentation_fn(train_sample['output']),
            }
        )
    for test_sample in sample['test']:
        new_sample['test'].append(
            {
                'input': augmentation_fn(test_sample['input']),
                'output': augmentation_fn(test_sample['output']),
            }
        )

    # other correct and cheap source of stochasticity
    random.shuffle(new_sample['train'])

    # choose 2-4 samples for inputs variability and as an augmentation
    n_train_samples = np.random.choice([2, 3, 4])
    new_sample['train'] = new_sample['train'][:n_train_samples]

    return new_sample


def apply_random_augmentation(sample) -> tuple[dict[str, list[dict[str, np.ndarray]]], tp.Callable]:
    augmentation_fn = choose_random_augmentation()

    new_sample = apply_augmentation_fn(sample, augmentation_fn)

    return new_sample, augmentation_fn
