from typing import Callable, Iterable, Optional, Union, List

import torch
import os
from glob import glob
from tqdm import tqdm
from TriangleMesh import TriangleMesh
 

class CustomObjNet(object):

    def __init__(self, basedir: str,
                 split: Optional[str] = 'train',
                 categories: Optional[Iterable] = ['bolt', 'nut', 'washer'],
                 transform: Optional[Callable] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        assert split.lower() in ['train', 'test']

        self.basedir = basedir
        self.transform = transform
        self.device = device
        self.categories = categories
        self.names = []
        self.filepaths = []
        self.cat_idxs = []

        if not os.path.exists(basedir):
            raise ValueError('CustomObjNet was not found at "{0}".'.format(basedir))

        available_categories = [p for p in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, p))]

        for cat_idx, category in enumerate(categories):
            assert category in available_categories, 'object class {0} not in list of available classes: {1}'.format(
                category, available_categories)

            cat_paths = glob(os.path.join(basedir, category, split.lower(), '*.obj'))
            self.cat_idxs += [cat_idx] * len(cat_paths)
            self.names += [os.path.splitext(os.path.basename(cp))[0] for cp in cat_paths]
            self.filepaths += cat_paths

        
        def __len__(self):
            return len(self.names)


        def __getitem__(self, index):
            """Returns the item at index idx. """
            category = torch.tensor(self.cat_idxs[index], dtype=torch.long, device=self.device)
            data = TriangleMesh.from_obj(self.filepaths[index])
            data.to(self.device)
            if self.transform:
                 data = self.transform(data)

            return data, category





