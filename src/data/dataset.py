import logging
import pickle

import numpy as np
import torch
from datasets import load_dataset

from .augment import apply_random_augmentation
from .compress import decode_sample
from models.tokenizer import TaskTokenizer

# EXCLUDE_KEYS = [
#     '0d3d703e',
#     '150deff5',
#     '1e32b0e9',
#     # '1f85a75f',
#     '22233c11',
#     '22eb0ac0',
#     '239be575',
#     # '25ff71a9',
#     '27a28665',
#     '28e73c20',
#     '2c608aff',
#     '2dee498d',
#     '32597951',
# ]
EXCLUDE_KEYS = [
    '484b58aa',  # -1 among tokens for some reason
]


class ARCDataset(torch.utils.data.Dataset):
    def _split_task_to_one_test_per_train(self, task):
        if len(task['test']) == 1:
            return [task]

        common_train = task['train']
        new_tasks = []
        for sample in task['test']:
            one_new_task = {'train': common_train, 'test': []}
            one_new_task['test'] = [
                {
                    'input': sample['input'],
                    'output': sample.get('output', None),
                }
            ]
            new_tasks.append(one_new_task)
        return new_tasks

    def __init__(self, named_tasks):
        # split tasks to one out per input
        keys = []
        tasks = []
        for key, task in named_tasks.items():
            splitted_tasks = self._split_task_to_one_test_per_train(task)

            tasks.extend(splitted_tasks)
            keys.extend([key for _ in splitted_tasks])

        self.keys = keys
        self.tasks = tasks

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        task = self.tasks[idx]

        return key, task


class HuggingFaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, num_samples: int | None = None, seed: int = 52):
        dataset = load_dataset(dataset_name, split='train')

        if num_samples is not None:
            if num_samples < len(dataset):
                generator = torch.Generator().manual_seed(seed)
                indices = torch.randperm(len(dataset), generator=generator).tolist()[:num_samples]
                dataset = torch.utils.data.Subset(dataset, indices)
            else:
                logging.warning(f'Requested {num_samples} samples from dataset, but only {len(dataset)} available')

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        key, encoded_sample = sample['key'], sample['sample']

        assert isinstance(encoded_sample, bytes), f'Encoded sample expected to be bytes, but got {encoded_sample}'
        task = decode_sample(encoded_sample)

        return key, task


class ReARCIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, all_generators_map, ub=1.0):
        self.all_keys = list(all_generators_map.keys())
        self.all_generators = list(all_generators_map.values())
        self.n_tasks = len(all_generators_map)

        self.ub = ub

    def generate_one(self):
        task_id = np.random.choice(self.n_tasks)
        key = self.all_keys[task_id]
        f = self.all_generators[task_id]

        train_samples = [f(0, self.ub) for _ in range(3)]
        test_samples = [f(0, self.ub)]
        sample = {'train': train_samples, 'test': test_samples}

        return key, sample

    def __iter__(self):
        while True:
            try:
                key, sample = self.generate_one()
            except IndexError:
                pass
            else:
                yield key, sample


class ReARCSampledDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        logger = logging.getLogger()

        logger.info('\tloading data')
        with open(datapath, 'rb') as f:
            data = pickle.load(f)
        logger.info('\tdone')

        self.data = data

        keys = []
        samples = []
        for task_key, task_samples in data.items():
            if task_key in EXCLUDE_KEYS:
                continue
            keys.extend([task_key for _ in range(len(task_samples))])
            samples.extend(task_samples)
        self.keys = keys
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.samples[idx]

        return key, sample


class AugmentDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that handles sample-level augmentations.
    """

    def __init__(self, dataset, tokenizer: TaskTokenizer, augment: bool = False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        key, sample = self.dataset[idx]
        if self.augment:
            augmented_sample, augmentation_fn = apply_random_augmentation(sample)
            augmented_key = f'{key}|{augmentation_fn.__name__}'
        else:
            augmented_sample = sample
            augmented_key = f'{key}|identity'

        result = {
            'key': key,
            'augmented_key': augmented_key,
            'raw_sample': sample,
            'augmented_sample': augmented_sample,
        }

        tokenized_data = self.tokenizer.tokenize(augmented_sample)
        result.update(tokenized_data)

        return result


def get_tokenized_datasets(
    raw_train_dataset, raw_all_val_datasets, tokenizer
) -> tuple[AugmentDataset, dict[str, AugmentDataset]]:
    train_dataset = AugmentDataset(raw_train_dataset, tokenizer, augment=True)

    all_val_datasets = {}
    for ds_name, val_or_test_dataset in raw_all_val_datasets.items():
        all_val_datasets[ds_name] = AugmentDataset(val_or_test_dataset, tokenizer, augment=False)

    return train_dataset, all_val_datasets
