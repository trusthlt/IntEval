import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configuration')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))

import copy
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler
import random
import numpy as np
from tqdm import tqdm

from config import Configuration
from processing.dataset import VioSet



class DataLoadECHR:
    def __init__(self, config_obj: Configuration, dataset: VioSet, parameters: dict):
        self.config = config_obj
        self.dataset = dataset
        self.tokenizer_update = False
        self.num_labels = len(self.dataset.lab2id)
        self.parameters = parameters
        self.class_weights = None
        self.tokenizer = self.set_tokenizer()

    def set_tokenizer(self) -> AutoTokenizer:
        """
        Method is used to set the tokenizer and adds special tokens (e.g., PAD)
        :return: tokenizer object
        """
        pretrained_path = self.config.model_names[self.parameters['model_path']]
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_path)
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer_update = True
        return tokenizer

    def get_text_from_tokens_wp_tokenizer(self, tokenized_text: list) -> str:
        """
        Method is used to collect words from the tokenized text for further processing )for bpe tokenization
        :param tokenized_text: list of tokens from the input text
        :return: string of the tokenized text, which joined by a space
        """
        # tokenizers as in bert
        if tokenized_text[0][:2] == '##':
            tokenized_text[0] = tokenized_text[0][2:]
        j = 1
        # tokenization splits long words into more little pieces, and the following lines collect those pieces
        # for further use in significance scoring
        while j < len(tokenized_text):
            if tokenized_text[j][:2] == '##':
                tokenized_text[j - 1] = tokenized_text[j - 1] + tokenized_text[j][2:]
                del tokenized_text[j]
            else:
                j += 1

        text = ' '.join(tokenized_text)
        return text

    # used for modernbert and legal longformer, which are not part of our classification process anymore
    def get_text_from_tokens_bpe_tokenizer(self, tokenized_text: list) -> str:
        """
        Method is used to collect words from the tokenized text for further processing )for bpe tokenization
        :param tokenized_text: list of tokens from the input text
        :return: string of the tokenized text, which joined by a space
        """
        # modern-bert tokenizers

        if tokenized_text[0][:1] == 'Ġ':
            tokenized_text[0] = tokenized_text[0][1:]

        j = 1
        # tokenization splits long words into more little pieces, and the following lines collect those pieces
        # for further use in significance scoring
        while j < len(tokenized_text):
            if tokenized_text[j][:1] != 'Ġ':
                tokenized_text[j - 1] = tokenized_text[j - 1] + tokenized_text[j]
                del tokenized_text[j]
            else:
                tokenized_text[j] = tokenized_text[j][1:]
                j += 1

        text = " ".join(tokenized_text)  # after augmentation applied

        return text

    def collate_function(self, batch: list, use_randomness: bool=False, hierarchical: bool=False) -> dict:
        """
        Extra processing is done on the batch data. If the text input is longer than the context size, then it is
        chunked into chunks. Each chunk is then further processed for further post-processing when apply the scoring
        :param hierarchical: boolean indicating whether to use hierarchical model (legal-bert, long-former) or not (modern-bert).
        :param batch: list of data to be collected in the batch
        :param use_randomness: window shift will be done randomly in the given range if it is set to True
        :return: dictionary with the following keys: input_ids, token_type_ids, attention_mask, labels, extra_info (meta information)
        """

        texts = list()
        label_list = list()
        extra_info = list()
        for sample in batch:
            texts.append(sample['text'])
            label_list.append(sample['labels'])
            extra_info.append(
                {
                    'itemid': sample['instance']['itemid'],
                    'text': sample['text']
                }
            )

        labels = np.array(label_list).astype('float32')
        all_splits_words = list()
        all_labels = list()
        count_snum = 0
        all_splits_num = list()
        for idx, text in enumerate(texts):
            # initially text is tokenized here
            tokenized_text = self.tokenizer.tokenize(text)
            upper_bound = max(min(130, len(tokenized_text) - 170), -20)
            random_shift = random.randint(-30, upper_bound)
            if random_shift < 0 or not use_randomness:
                random_shift = 0

            tokenized_text = tokenized_text[random_shift:]

            if self.parameters['model_path'] == 'legal_bert':
                text = self.get_text_from_tokens_wp_tokenizer(tokenized_text)
                words, num_splits, splits_counts = self.split_sample_legal_bert(text)

            else:
                text = self.get_text_from_tokens_bpe_tokenizer(tokenized_text)
                words, num_splits, splits_counts = self.split_sample_modern_bert(text)

            # a sample is chosen out of chunks to represent this specific datapoint
            if not hierarchical:
                sample = random.randrange(0, num_splits)

                splits_words = [''.join(w['tokens']) for w in words if sample in w['splits']]
                all_splits_words.append(splits_words)
                all_splits_num.append(1)
            else:
                for sample in range(num_splits):
                    count_snum += 1
                    splits_words = [''.join(w['tokens']) for w in words if sample in w['splits']]
                    all_splits_words.append(splits_words)
                all_splits_num.append(num_splits)


            extra_info[idx].update({'num_splits': num_splits})
            all_labels.append(labels[idx])

        tokenized_inputs = self.tokenizer(
            all_splits_words, return_tensors='pt', max_length=self.parameters["context_size"],
            truncation=True, is_split_into_words=True, padding='max_length'
        )

        tokenized_inputs['labels'] = torch.LongTensor(all_labels)
        tokenized_inputs['extra_info'] = extra_info
        tokenized_inputs['itemid'] = [each['itemid'] for each in extra_info]

        return tokenized_inputs

    def split_sample_modern_bert(self, updated_text: str, separator: str="Ġ", additionally_extend: bool=True) -> tuple:
        """
        Method applies splitting to the sample, when the Byte Level (BPE) tokenization is used.
        :param updated_text: text to be split
        :param separator: string to identify the separator
        :param additionally_extend: boolean indicating whether to add additional tokens
        :return: tuple of:
            - list of words (not tokens) and information about how many tokens they have, which splits they are in;
            - number of splits given the text;
            - list of token size of splits.
        """
        tokenized_text = self.tokenizer.tokenize(updated_text)
        if self.parameters['model_path'] == 'legal_longformer':
            num_splits = int(np.ceil(len(tokenized_text) / 3900))
        else:
            num_splits = int(np.ceil(len(tokenized_text) / 7900))

        words = list()
        current_word_tokens = list()
        num_tokens_per_split = {idx: 0 for idx in range(0, num_splits)}
        current_split = 0
        if self.parameters['model_path'] == 'legal_longformer':

            tokens_per_split = self.parameters['context_size'] - 100
        else:
            tokens_per_split = self.parameters['context_size'] - 200

        for token in tokenized_text:
            if token[:1] == separator:  # if the current token is a new word
                if len(current_word_tokens):  # if this is true, then we switched to the new token already, and we need to save the previous one
                    info_dict = {
                        'tokens': current_word_tokens,  # tokens of the current word
                        'num_tokens': len(current_word_tokens),
                        'word': ''.join(current_word_tokens),
                        'splits': [current_split]
                    }
                    words.append(info_dict)
                    num_tokens_per_split[current_split] += len(current_word_tokens)

                    if num_tokens_per_split[current_split] >= tokens_per_split:
                        # if number of tokens in this split is higher than the pre-defined threshold, new split will be created
                        current_split += 1
                        if self.parameters['model_path'] == 'legal_longformer':
                            tokens_per_split = self.parameters['context_size'] - 100
                        else:
                            tokens_per_split = self.parameters['context_size'] - 200
                current_word_tokens = [token[1:]]

            else:
                current_word_tokens.append(token)

        words.append(
            {
                "tokens": current_word_tokens,
                "num_tokens": len(current_word_tokens),
                "word": "".join(current_word_tokens),
                "splits": [current_split]
            }
        )
        if self.parameters['model_path'] == 'legal_longformer':
            max_size = self.parameters['context_size'] - 100
        else:
            max_size = self.parameters['context_size'] - 200
        num_tokens_per_split[current_split] += len(current_word_tokens)

        if num_splits > 1:
            # if number of tokens per split is very little (less than 100 tokens) and combining it with the previous one
            # does not exceed the maximum size allowed for split, then we merge the last one to the one previous.
            if num_tokens_per_split[num_splits - 1] <= 100 and sum(
                    list(num_tokens_per_split.values())[-2:]) <= max_size:
                for word in words:
                    word['splits'] = word['splits'] if num_splits - 1 not in word['splits'] else [num_splits - 2]
                num_splits -= 1
                num_tokens_per_split[num_splits - 1] += num_tokens_per_split[num_splits]
                del num_tokens_per_split[num_splits]

        words, num_tokens_per_split = self.rearrange_splits(
            words,
            num_tokens_per_split,
            max_size,
            additionally_extend=additionally_extend
        )
        return words, num_splits, num_tokens_per_split


    def split_sample_legal_bert(self, updated_text: str, separator: str= '##', additionally_extend: bool=True):
        """
        Method applies splitting to the sample, when the Word Piece (WP) tokenization is used.
        :param updated_text: text to be split
        :param separator: string to identify the separator
        :param additionally_extend: boolean indicating whether to add additional tokens
        :return: tuple of:
            - list of words (not tokens) and information about how many tokens they have, which splits they are in;
            - number of splits given the text;
            - list of token size of splits.
        """
        tokenized_text = self.tokenizer.tokenize(updated_text)
        num_splits = int(np.ceil(len(tokenized_text)/400))

        words = list()
        current_word_tokens = list()
        num_tokens_per_split = {idx: 0 for idx in range(0, num_splits)}
        current_split = 0
        tokens_per_split = self.parameters['context_size'] - 12

        for token in tokenized_text:
            if token[:2] != separator: # if the current token is a new word
                if len(current_word_tokens): # if this is true, then we switched to the new token already, and we need to save the previous one
                    info_dict = {
                        'tokens': current_word_tokens, # tokens of the
                        'num_tokens': len(current_word_tokens),
                        'word': ''.join(current_word_tokens),
                        'splits': [current_split]
                    }
                    words.append(info_dict)
                    num_tokens_per_split[current_split] += len(current_word_tokens)

                    if num_tokens_per_split[current_split] >= tokens_per_split: # if number of tokens in this split is higher than the pre-defined threshold, new split will be created
                        current_split += 1
                        tokens_per_split = 400

                current_word_tokens = [token]

            else:
                current_word_tokens.append(token[2:])

        words.append(
            {
                "tokens": current_word_tokens,
                "num_tokens": len(current_word_tokens),
                "word": "".join(current_word_tokens),
                "splits": [current_split]
             }
        )
        max_size = self.parameters['context_size'] - 2
        num_tokens_per_split[current_split] += len(current_word_tokens)

        if num_splits > 1:
            # if number of tokens per split is very little (less than 100 tokens) and combining it with the previous one
            # does not exceed the maximum size allowed for split, then we merge the last one to the one previous.
            if num_tokens_per_split[num_splits-1] <= 100 and sum(list(num_tokens_per_split.values())[-2:]) <= max_size:
                for word in words:
                    word['splits'] = word['splits'] if num_splits - 1 not in word['splits'] else [num_splits - 2]
                num_splits -= 1
                num_tokens_per_split[num_splits - 1] += num_tokens_per_split[num_splits]
                del num_tokens_per_split[num_splits]

        words, num_tokens_per_split = self.rearrange_splits(
            words,
            num_tokens_per_split,
            max_size,
            additionally_extend=additionally_extend
        )
        return words, num_splits, num_tokens_per_split

    def rearrange_splits(self, words_input: list, num_tok_input: list, max_size: int,
                         additionally_extend: bool=True) -> tuple:
        """
        Method is used to rearranging splits for overlapping and having the similar sizes
        :param words_input: list of words
        :param num_tok_input: list of number of tokens per split
        :param max_size: maximum size of the split
        :param additionally_extend: boolean indicating whether to add additional tokens
        :return: tuple of:
            - list of words (not tokens) and information about how many tokens they have after rearranging
            - number of tokens per split after rearranging
        """
        words = copy.deepcopy(words_input)
        num_tokens_per_split = copy.deepcopy(num_tok_input)
        threshold_size = self.parameters['context_size'] - 32
        tokens_since_last_reset = 100
        current_split = len(num_tokens_per_split)
        for i in range(len(words)):
            current_word = words[-(i + 1)]
            can_extend = (
                    current_split in num_tokens_per_split
                    and num_tokens_per_split[current_split] < threshold_size
                    and additionally_extend
            )
            fits_in_current = (
                    num_tokens_per_split[current_split] + current_word["num_tokens"] <= max_size
            ) if current_split < len(num_tokens_per_split) else False
            if (
                    (tokens_since_last_reset < 100 or can_extend)
                    and current_split not in current_word["splits"]
                    and fits_in_current
            ):
                current_word["splits"].append(current_split)
                tokens_since_last_reset += current_word["num_tokens"]
                num_tokens_per_split[current_split]+= current_word["num_tokens"]
            elif tokens_since_last_reset >= 100 or not fits_in_current:
                current_split = current_word["splits"][0]
                tokens_since_last_reset = 0

        return words, num_tokens_per_split

    def __getitem__(self, ds_name: str) -> DataLoader:
        """
        Method gets dataloader according to the given split name
        :param ds_name: split information
        :return: data loader for the specific split
        """
        # we set randomness to False, because we will use either 8k model for specific dataset, or l-bert in
        # hierarchical setting, so models will see all data

        self.dataset.set_split(ds_name)

        collate_fn = (lambda x: self.collate_function(x, use_randomness=False, hierarchical=True))
        data_loader = DataLoader(
            dataset=self.dataset,
            collate_fn=collate_fn,
            batch_size=self.parameters['batch_size'],
            shuffle=True
        )

        return data_loader
