import os
import pickle
import pandas as pd
import ast


class ProcessData:
    def __init__(self, parameters: dict, split_info: str):
        self.parameters = parameters
        self.split = split_info
        self.lab2id = {'positive': 1, 'negative': 0}
        self.dataset = self.__main__()

    def read_data(self, split_info: str) -> dict:
        """
        Using the split information, interpretability subset of ECtHR dataset is collected from the given dataset
        :param split_info: string to specify the split information (e.g., train, dev, test)
        :return: dictionary that contains ECtHR dataset
        """
        if self.parameters['article_id'] == 'all':
            dataset_path = os.path.join(self.parameters['dataset_path'], 'interpretability_dataset.csv')
        else:
            dataset_path = os.path.join(self.parameters['dataset_path'], f'interpretability_dataset_art_{self.parameters["article_id"]}.csv')
        data = pd.read_csv(dataset_path)
        data['text'] = data['text'].apply(ast.literal_eval)
        data['raw_text'] = data['text'].apply(lambda x: ' '.join(x))
        data['articles'] = data['article'].apply(ast.literal_eval)
        subframe = data[data['split'] == split_info]
        dataset = subframe.to_dict(orient='records')
        return dataset

    def __main__(self):
        """
        Combines all splits in one document to make the usage simpler
        :return: dataset of the given split
        """
        if self.parameters['article_id'] == 'all':
            ds_path = os.path.join(self.parameters['dataset_path'], 'binary_echr.pickle')
        else:

            ds_path = os.path.join(self.parameters['dataset_path'], f'data_binary_{self.parameters["article_id"]}.pickle')
        dataset = dict()
        if not os.path.exists(ds_path):
            for split in ['train', 'val', 'test']:
                dataset[split] = self.read_data(split)
            with open(ds_path, 'wb') as f:
                pickle.dump(dataset, f)
        with open(ds_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset[self.split]

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)
