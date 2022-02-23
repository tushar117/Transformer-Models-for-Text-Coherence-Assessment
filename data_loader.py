import torch
import json
import linecache
from torch.utils.data import DataLoader, TensorDataset, Dataset


class TextDataset(Dataset):
    def __init__(self, filename, total_entries, regression):
        self.filename = filename
        self.total_entries = total_entries
        self.float_label = regression
    
    def _add_if_present(self, key, json_data, return_list, dtype):
        if key in json_data:
            return_list.append(torch.tensor(json_data[key], dtype=dtype))

    def preprocess(self, json_data):
        return_list = []

        # d_id is added for identifying the task in multi-task-learning setup 
        key_order = ['d_id', 'doc_a', 'doc_a_mask', 'doc_a_facts', 'doc_a_facts_mask', 'doc_a_facts_count', 
                    'doc_b', 'doc_b_mask', 'doc_b_facts', 'doc_b_facts_mask', 'doc_b_facts_count', 
                    'label']

        for key in key_order:
            dtype = torch.long
            if key == 'label' and self.float_label:
                dtype = torch.float
            self._add_if_present(key, json_data, return_list, dtype)

        return tuple(return_list)

    def line_mapper(self, line):
        json_data = json.loads(line.strip())
        datasets = self.preprocess(json_data)
        return datasets

    def __getitem__(self, idx):
        line = linecache.getline(self.filename, idx+1)
        return self.line_mapper(line)

    def __len__(self):
        return self.total_entries

def get_dataset_loaders(filename, count, batch_size=8, num_threads=0, regression=False):
    dataset = TextDataset(filename, count, regression)
    input_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_threads)
    return input_dataloader
