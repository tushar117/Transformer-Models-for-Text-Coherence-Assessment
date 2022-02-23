import torch
import json
import os, sys
import linecache
from torch.utils.data import DataLoader, TensorDataset, Dataset

# required to access the python modules present in project directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# now we can import the all modules present in project folder
from utils.common import load_file


class TextDataset(Dataset):
    def __init__(self, filename, float_label):
        self.filename = filename
        self.dataset = load_file(filename)
        self.float_label = float_label
    
    def _add_if_present(self, key, json_data, return_list, dtype):
        if key in json_data:
            return_list.append(torch.tensor(json_data[key], dtype=dtype))

    def preprocess(self, json_data):
        return_list = []

        # prompt_id for identifying different prompt types
        # d_id is added for identifying the task in multi-task-learning setup 
        key_order = ['prompt_id', 'd_id', 'essay_id', 'doc_a', 'doc_a_mask', 'doc_a_facts', 'doc_a_facts_mask', 'doc_a_facts_count', 
                    'doc_b', 'doc_b_mask', 'doc_b_facts', 'doc_b_facts_mask', 'doc_b_facts_count', 'coherence_vector',  
                    'label']

        for key in key_order:
            dtype = torch.long
            if key == 'label' and self.float_label or key == "coherence_vector":
                dtype = torch.float
            self._add_if_present(key, json_data, return_list, dtype)

        return tuple(return_list)

    def __getitem__(self, idx):
        data_instance = self.dataset[idx]
        return self.preprocess(data_instance)

    def __len__(self):
        return len(self.dataset)

def get_dataset_loaders(filename, batch_size=8, num_threads=0, float_label=True):
    dataset = TextDataset(filename, float_label=float_label)
    input_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_threads)
    return input_dataloader
