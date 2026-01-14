import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from processing.process_data import ProcessData

class VioSet(Dataset):
    def __init__(self, parameters, config_obj, split):
        self.parameters = parameters
        self.split = None
        self.process_obj = None
        self.lab2id = None
        self.config_obj = config_obj
        self.tokenizer = AutoTokenizer.from_pretrained(self.config_obj.model_names[parameters['model_path']])
        self.set_split(split)

    def set_split(self, split: str) -> None:
        """
        Method is used to set the split of the dataset.
        :param split: string object can be train, dev or test.
        :return: None
        """
        self.split = split
        self.process_obj = ProcessData(self.parameters, self.split)
        self.lab2id = self.process_obj.lab2id

    def get_sample(self, idx: int) -> dict:
        """
        Given the data index, method returns the sample
        :param idx: index of the sample in the dataset
        :return: dictionary that contains the text, label index, and processed relevant information per sample
        """
        sample = self.process_obj[idx]
        return {
            'text': sample['raw_text'],
            'labels': self.lab2id[sample['labels']],
            'instance': sample
        }

    def __len__(self) -> int:
        """
        Length of the dataset.
        :return: length of the dataset
        """
        return len(self.process_obj.dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Given the data index, method returns the sample
        :param idx: index of the sample in the dataset
        :return: dictionary contains all relevant information per sample
        """
        if self.parameters['model_path'] != 'legal_bert':
            if self.parameters['hierarchical']:
                return self.get_sample(idx)
            else:
                sample = self.get_sample(idx)

                data = self.tokenizer(sample['text'], return_tensors='pt', padding='max_length',
                                      max_length=self.tokenizer.model_max_length)
                data.update({'labels': sample['labels'], 'itemid': sample['instance']['itemid'],
                             'token_type_ids': torch.zeros_like(data['input_ids']), 'extra_info': {'itemid': sample['instance']['itemid'], 'text': sample['text']}})

                return data
        else:
            return self.get_sample(idx)



