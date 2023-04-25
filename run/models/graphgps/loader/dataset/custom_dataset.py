from typing import Optional, Callable, List

import os
import glob
import os.path as osp

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar, extract_zip)
from torch_geometric.utils import remove_isolated_nodes


class CustomDataset(InMemoryDataset):
    r"""My custom dataset
    """

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ""

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'split_dict.pt']

    def download(self):
        return

    def _generate_data_list(self):
        raise NotImplemented("This method is implemented outside")

    def process(self):
        data_list = self._generate_data_list()
        self.data_list = data_list
        split_dict = {'train': [], 'valid': [], 'test': []}

        split_dict['train'].append([i for i in range(0, 23)])
        split_dict['valid'].append([i for i in range(23, 33)])
        split_dict['test'].append([i for i in range(33, LEN_MODELS)])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])

    def get_idx_split(self):
        return torch.load(self.processed_paths[1])