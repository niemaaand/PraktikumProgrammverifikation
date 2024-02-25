import torch
from torch.utils.data import Dataset

from src.vnnlib.properties import Properties


class RandomDataDataset(Dataset):
    def __init__(self, size, n_samples, properties: Properties):
        super(RandomDataDataset).__init__()
        self._properties = properties
        self.random_dataset = self._create_random_dataset(n_samples, size)

    def _create_random_dataset(self, n_samples, size):
        samples = []
        for s in range(n_samples):
            samples.append((self._properties.calc_random_input_vector(size), torch.zeros(size[0])))

        return samples

    def __len__(self):
        return len(self.random_dataset)

    def __getitem__(self, index):
        return self.random_dataset[index][0], self.random_dataset[index][1]
