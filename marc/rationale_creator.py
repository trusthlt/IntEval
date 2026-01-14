import os.path
import copy
import torch
import warnings
import numpy as np
import sys
from tqdm import tqdm
import pickle
import time
import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers.tokenization_utils_base import BatchEncoding

class RationaleCreateObj:
    def __init__(self, parameters_dict, config_obj, dataset_obj, dataloader_obj, trainer_obj, num_optimization_steps=900,
                 random_change_percentage=0.05, weights_loss_factor=1, sigma_loss_factor=1.2):
        self.parameters = parameters_dict
        self.configuration = config_obj
        self.dataset = dataset_obj
        self.dataloader = dataloader_obj
        self.significance_path = None
        self.trainer = trainer_obj
        self.model = None
        self.set_model()

        self.idx_name = None
        self.num_optimization_steps = num_optimization_steps
        self.random_change_percentage = random_change_percentage
        self.weights_loss_factor = weights_loss_factor
        self.sigma_loss_factor = sigma_loss_factor
        self.limit_ids = list()

    def set_model(self) -> None:
        """
        Method sets the model of the experiment
        :return: None
        """
        self.trainer.test_model()
        self.model = self.trainer.model

    @staticmethod
    def check_dir(directory: str) -> None:
        """
        Method checks if the directory exists
        :param directory: path to the directory to check
        :return: None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def check_outputs(self):
        result_folder = os.path.join(self.significance_path, 'results', 'meta_data_marc')
        priority_list = [
            '001-120512', '001-201325', '001-79415', '001-92582', '001-94578',
            '001-145215', '001-175482', '001-191714', '001-219778', '001-228385'
        ]

        for sample in self.dataset:
            doc_id = sample["instance"]["itemid"]
            if sample["instance"]["itemid"] in priority_list:
                file_path = os.path.join(result_folder, f'sample_{sample["instance"]["itemid"]}.pickle')
                data = pickle.load(open(file_path, 'rb'))
                print(doc_id, data['label'])


    def rationale_creation_process(self, split: str='test', set_priority=True):
        """
        Main function which controls rationale creation process
        :param split: split of the dataset to be used for extraction
        :return: None
        """
        if split not in ['train', 'val', 'test']:
            raise NotImplementedError(f'Split should be one of [\"train\", \"val\", \"split\"]')

        self.check_dir(self.significance_path)

        progress_bar = tqdm(
            iterable=enumerate(self.dataset),
            desc=f"{self.parameters['process_name']}: Generating rationale for sample None / None",
            total=self.dataset.__len__(),
            position=0,
            leave=True,
            file=sys.stdout
        )
        result_folder = os.path.join(self.significance_path, 'results', 'meta_data_marc')
        self.check_dir(result_folder)
        priority_list = [
            '001-120512', '001-201325', '001-79415', '001-92582', '001-94578',
            '001-145215', '001-175482', '001-191714', '001-219778', '001-228385'
        ]
        for idx, sample in progress_bar:

            if set_priority and sample['instance']['itemid'] not in priority_list:
                self.parameters['limit_eval'] = False
                continue

            file_path = os.path.join(result_folder, f'sample_{sample["instance"]["itemid"]}.pickle')
            if self.parameters['limit_eval'] and self.parameters['limit_eval'] == idx:
                print(f'INFO: Rationale creation was intended to be limited with {self.parameters["limit_eval"]}')
                break
            self.limit_ids.append(sample['instance']['itemid'])
            if not os.path.exists(file_path):
                desc_info = f'{self.parameters["process_name"]}: Generating rationale for sample {sample["instance"][self.idx_name]}'
                self.rationale_creation_main(sample, file_path, progress_bar, desc_info)

    def collect_req_info(self, text):

        words, num_splits, split_word_counts = self.dataloader.split_sample_legal_bert(text)
        max_split_length = max(split_word_counts.values())
        split_words, all_word_tokens_counts, split_word_tokens_counts, num_pads_per_split = self.data_prep_per_creation(
            words, num_splits, split_word_counts, max_split_length)

        sample_tokenized = self.dataloader.tokenizer(split_words, return_tensors='pt', truncation=False,
                                                     is_split_into_words=True)

        # --------------- ADDING QUERY MASK FOR ISR SETUP --------------------- #
        # the following 2 lines are taken from ISR dataholder's encodeplusplus part, since it will be needed
        sample_tokenized["query_mask"] = sample_tokenized["attention_mask"].clone()
        split_info = {
            'words': words,
            'num_splits': num_splits,
            'split_word_counts': split_word_counts,
        }
        return sample_tokenized, split_info

    @staticmethod
    def data_prep_per_creation(words: list, num_splits: int, split_word_counts: dict, max_split_length: int) -> tuple:
        """
        Method prepares the data for rationale creation and will also be used for mapping weights back to original words
        :param words: list of words, where each word is a dictionary with keys of tokens (tokens that make this word),
                      num_tokens (number of tokens in the word [longer words can contain several tokens], integer),
                      word (word itself as a string), splits (list shows the splits that the word belongs to)
        :param num_splits: integer specifying the number of the splits that the sample was chunked into
        :param split_word_counts: dictionary with keys of split indexes where the values are number of words in chunk
        :param max_split_length: maximum number of words in all splits
        :return: tuple of following elements:
                split_words: list of lists, where each list contains the words belonging to each split (order is preserved)
                all_word_tokens_counts: list of token numbers for words in whole document
                split_word_tokens_counts: list of lists, where each list contains the number of tokens per word:
                e.g., if 15th word of 2nd split has 3 tokens, then split_word_tokens_counts[1][14] will be 3, where
                split_word[1][14] will correspond to this very word; notice special tokens are numbered as -1;
                num_pads_per_split: list of number of padding was added to the split, where each number corresponds to
                each split
        """
        split_word_tokens_counts = [[-1] for _ in range(num_splits)]
        split_words = [list() for _ in range(num_splits)]

        # 1. collects words per split and number of token per each word (tracking)
        for word in words:
            for split in word['splits']:
                split_word_tokens_counts[split].append(word['num_tokens'])
                split_words[split].append(word['word'])
        num_pads_per_split = list()
        # 2. padding splits to the length of the split with the maximum length
        for idx in range(num_splits):
            diff = max_split_length - split_word_counts[idx]
            if diff > 0:
                split_word_tokens_counts[idx] += [-1] * diff  # -1 for special tokens [PAD], [CLS] -> [PAD] here
                split_words[idx] += ['[PAD]'] * diff
            num_pads_per_split.append(diff)

        all_word_tokens_counts = list()
        # 3. collect all words in the splits (notice, overlapping is not engineered here)
        for i in range(num_splits):
            split_word_tokens_counts[i].append(-1)
            all_word_tokens_counts = all_word_tokens_counts + split_word_tokens_counts[i]

        return split_words, all_word_tokens_counts, split_word_tokens_counts, num_pads_per_split

    def get_embeddings(self, tokenized: BatchEncoding, uninformative: BatchEncoding) -> tuple:
        """
        Method gets embeddings for a batch of tokenized documents and uninformative input
        :param tokenized: tokenized input text
        :param uninformative: uninformative input text, all tokens of which are padding element [PAD]
        :return: tuple of input embeddings and uninformative embeddings
        """

        embeddings_sample = self.model.core_model.model.embeddings(tokenized["input_ids"].to(self.configuration.device))
        embeddings_uninformative = self.model.core_model.model.embeddings(uninformative["input_ids"].to(self.configuration.device))

        return embeddings_sample, embeddings_uninformative

    def learnable_params(self, all_word_tokens_counts: list) -> tuple:
        """
        Method initiates learnable parameters for the optimization process
        :param all_word_tokens_counts: list of number, where each number represents how many tokens of each word
        :return: tuple of lists, where each list contains initial weight and sigma per word (not tokens)
        """
        weights = list()
        sigmas = list()
        for r in all_word_tokens_counts:
            if r != -1: # i.e., not PAD or CLS
                weight = torch.tensor([1.2], requires_grad=True, device=self.configuration.device, dtype=torch.float32)
                sigma = torch.tensor([2.0], requires_grad=True, device=self.configuration.device, dtype=torch.float32)
            else:
                weight = torch.tensor([-20], requires_grad=False, device=self.configuration.device, dtype=torch.float32)
                sigma = torch.tensor([0.01], requires_grad=False, device=self.configuration.device, dtype=torch.float32)
            weights.append(weight)
            sigmas.append(sigma)
        return weights, sigmas

    def compute_masked_loss(self, mask_predictions: float, label: int) -> torch.Tensor:
        """
        Compute masked loss with respect to the prediction (because our modification of the process makes the rationales
        based output converge to the prediction)

        Notice: If you provide complement mask_predictions (as in main function), it will compute the loss accordingly

        :param mask_predictions: output of the model based on the masked input
        :param label: prediction label
        :return: loss based on the masked input (Tensor)
        """
        comp_mask_predictions = 1 - mask_predictions
        comp_label = 1 - label
        loss = -comp_label * torch.log(comp_mask_predictions) - label * torch.log(mask_predictions)

        return torch.squeeze(loss, dim=-1)

    def compute_distance(self, current_num_words: int, sigmas: torch.Tensor) -> torch.Tensor:
        """
        Method computes the distance among all words in the provided split
        :param current_num_words: Number of words in current split
        :param sigmas: sigmas for the corresponding split
        :return: Tensor: distance for each words in split with respect to all words in current split
        """
        count_vector = torch.arange(start=0, end=current_num_words, step=1).to(self.configuration.device)
        positional_tensor = count_vector.repeat((current_num_words, 1))
        positional_vector = torch.unsqueeze(count_vector, -1)
        difference = positional_tensor - positional_vector

        distance = torch.square(difference) / torch.square(torch.unsqueeze(sigmas, dim=-1))

        return distance

    def initiate_masks(self, split_set: list, prev_split_end: int, split_word_tokens_counts: list, sigmas: list,
                       weights: list, num_pads_per_split: list) -> tuple:
        """
        Information about initiating masks:
        info if needed
        The attention mask shape is defined as (2 * len(split_set), max_split_length + 2) to accommodate the embeddings
        for both the masked and complement masked versions of the input data.
        Here's a breakdown of the dimensions:
            -2 * len(split_set): The factor of 2 accounts for both the masked and complement masked embeddings.
            -Each split in split_set will have two versions: one with the mask applied and one with the complement
            of the mask applied.
            -max_split_length + 2: This accounts for the maximum length of the splits plus two additional tokens
            (typically for special tokens like [CLS] and [SEP]).
        This ensures that the attention mask can cover all tokens in both the masked and complement masked embeddings for each split.

        :param split_set: indexes of the split set that are processed now (since, not all of them are processed directly)
        :param prev_split_end: integer representing the end of the current split.
        :param split_word_tokens_counts: list of lists, where each list contains number of tokens of each word in split
        :param sigmas: list of tensors, where each tensor represents the weight of each split
        :param weights: list of tensors, where each tensor represents the weight of each split
        :param num_pads_per_split: list of numbers, where each number represents how many paddings for each split
        :return: tuple of following elements:
            mask_tensors: tensor, where each axis corresponds to a split, and it includes repeated values for words that
            has multiple tokens
            all_individual_mask_values: list of tensors, where each tensor represent the masks of each split
            (w/o special tokens)
            all_individual_sigmas: sigma values for each split (list)
            prev_split_end: integer representing the end of the current split.
        """
        mask_tensors = list()
        all_individual_mask_values = list()
        all_individual_sigmas = list()
        for j in split_set:
            current_num_words = len(split_word_tokens_counts[j])
            sigmas_tensor = torch.cat(sigmas[prev_split_end: prev_split_end + current_num_words])
            all_individual_sigmas.append(sigmas_tensor)
            weights_tensor = torch.cat(weights[prev_split_end: prev_split_end + current_num_words])
            prev_split_end = prev_split_end + current_num_words
            distance = self.compute_distance(current_num_words, sigmas_tensor)
            mask_values = torch.sigmoid(
                (torch.exp(-distance) * torch.unsqueeze(weights_tensor, dim=-1)).sum(0)
            )
            all_individual_mask_values.append(mask_values[1:-(1 + num_pads_per_split[j])])

            # repeats each value in mask_values according to the corresponding count in split_word_tokens_counts[j].
            # This ensures that each mask value is replicated for the number of tokens in each word, so that the
            # mask can be applied to all tokens of a word, not just the word itself. This is necessary because some
            # words may be tokenized into multiple tokens, and the mask needs to cover all of them.

            mask_values_repeated = [mask_values[k].repeat(split_word_tokens_counts[j][k]) for k in
                                    range(len(split_word_tokens_counts[j]))]
            mask_values_repeated = torch.unsqueeze(
                torch.unsqueeze(torch.cat(mask_values_repeated, dim=0), dim=-1), dim=0)
            mask_tensors.append(mask_values_repeated)
        mask_tensors = torch.squeeze(torch.stack(mask_tensors, dim=0), dim=1)
        mask_tensors = self.create_noise(mask_tensors)

        return mask_tensors, all_individual_mask_values, all_individual_sigmas, prev_split_end

    def create_noise(self, masked_input: torch.Tensor) -> torch.Tensor:
        """
        Method add noise to masked input
        :param masked_input: tensor, where each axis corresponds to a split, and it includes repeated values for words
        that has multiple tokens
        :return: masked input with added noise
        """
        ones = torch.ones_like(masked_input, device=self.configuration.device, dtype=torch.float32)
        d_1 = (torch.empty_like(masked_input,
                                device=self.configuration.device).uniform_() > self.random_change_percentage).type(
            torch.float32)
        # elements that are 1 (as a result of the comparison) will be set to 0 in mask tensors
        d_2 = (torch.empty_like(masked_input,
                                device=self.configuration.device).uniform_() > self.random_change_percentage).type(
            torch.float32)
        # elements that are 1 (as a result of the comparison) will be set to 1 in mask tensors
        both = (1 - d_1) * (1 - d_2)
        # elements that are chosen neither by d1 nor by d2 (i.e, set to 0 in both cases), will be 1 in both. This
        # will indicate which elements should remain as before

        d_1 = d_1 + both * ones
        d_2 = d_2 + both * ones
        # Applying selections by keeping the "both" values unchanged
        return masked_input * d_1 * d_2 + ones * (1 - d_2)

    def rationale_creation_main(self, datapoint : tuple, filepath: str, progress_bar: tqdm,
                                description: str, max_num_par_splits: int=3) -> None:
        """
        Method is used for rationale creation, which includes all the steps (follow the numbers :) )
        :param datapoint: tuple that represent a sample
        :param filepath: path to save the rationale
        :param progress_bar: tqdm progress bar
        :param description: description of the progress bar to be updated
        :param max_num_par_splits: maximum number of splits can be processed
        :return: None
        """

        label = datapoint['labels']
        # 1. chunk the input document, since it is too long in the legal domain and get the maximum length among them
        words, num_splits, split_word_counts = self.dataloader.split_sample_legal_bert(datapoint['text'])
        extra_info = [{'num_splits': num_splits}]
        max_split_length = max(split_word_counts.values())

        # 2. collect required information for the rationale creation (detailed info in docstring of th efunction)
        split_words, all_word_tokens_counts, split_word_tokens_counts, num_pads_per_split = (
            self.data_prep_per_creation(words, num_splits, split_word_counts, max_split_length))

        # 3. tokenize split sample and uninformative split
        sample_tokenized = self.dataloader.tokenizer(
            split_words, return_tensors='pt', truncation=False, is_split_into_words=True
        )
        uninformative_input = self.dataloader.tokenizer(
            [("[PAD] " * max_split_length)[:-1]] * num_splits, return_tensors='pt', truncation=False
        )

        # 4. collect embeddings for input sample and uninformative input
        embeddings_sample, embeddings_uninformative = self.get_embeddings(sample_tokenized, uninformative_input)

        # 5. initiate weight and sigmas
        weights, sigmas = self.learnable_params(all_word_tokens_counts=all_word_tokens_counts)

        # 6. initiate the optimizer
        optimizer = torch.optim.AdamW(weights + sigmas, lr=3e-2)
        progress_bar.set_description(f'{description} => Model, Data and Optimizer was set up for the given datapoint')

        num_parameters = list()
        for k, r in enumerate(split_word_tokens_counts):
            current_params = list()
            for idx, val in enumerate(r):
                if val == -1:
                    split_word_tokens_counts[k][idx] = 1
                    continue
                current_params.append(val)
            num_parameters.append(len(current_params))

        # 7. setting the probability function for the output of the model (how to process the logits?)
        probability_func = (lambda x: torch.softmax(x, dim=-1))

        # 8. we use prediction values to read rationales not the ground truth
        if not self.parameters['access_to_gt']:
            input_batch = {
                'input_ids': None,
                'attention_mask': torch.ones((num_splits, max_split_length + 2)).to(self.configuration.device),
                'token_type_ids': None,
                'inputs_embeds': embeddings_sample,
                'retain_gradient': False,
                'extra_info': extra_info
            }
            model_output, _ = self.model(**input_batch)

            out = probability_func(model_output)
            label = out.argmax(dim=-1).item()

        split_indices = [list(range(num_splits))]


        final_weights = dict()
        prev_split_end_prev_set = 0

        # 9. optimization is done for each split
        for sets in split_indices:

            progress_bar.set_description(
                f'{description} => {sets} out of {len(split_indices)} different sets')

            # Initiate the list to track the average mask value to check for stop conditions
            last_mask_means = [1 for _ in sets]
            attention_mask = torch.ones((2 * len(sets), max_split_length + 2)).to(self.configuration.device)
            # 10: Optimization loop, max number of steps is set to 900
            for optimization_step in range(self.num_optimization_steps):

                prev_split_end = prev_split_end_prev_set
                classification_loss = torch.zeros((1,), device=self.configuration.device, dtype=torch.float32)
                all_predictions = list()
                # 10.1. initiating the masks
                mask_tensors, individual_masks, individual_sigmas, prev_split_end = self.initiate_masks(sets, prev_split_end, split_word_tokens_counts, sigmas, weights, num_pads_per_split)
                # 10.2. masking input and uninformative input (mask_tensors corresponds to lambda in the equations)
                masked_embeddings = embeddings_sample[sets] * mask_tensors + embeddings_uninformative[sets] * (1 - mask_tensors)
                # 10.3. computing the complement mask embeddings
                complement_masked_embeddings = embeddings_sample[sets] * (1 - mask_tensors) + embeddings_uninformative[sets] * mask_tensors

                # 10.4 predicting the outcome based on masked embeddings and complement masked embeddings
                embeddings = torch.cat([masked_embeddings, complement_masked_embeddings], dim=0)
                embeddings += torch.empty_like(embeddings, device=self.configuration.device).normal_(mean=0.0, std=0.03)
                # in this current setup we send a batch in a way that, batch has 2 objects: Masked and complement, each has same dimension.
                # Since, we implement hierarchical setup, each of them processed as a document (all chunks are processed through the model) and
                # output becomes 2 (num of docs -> masked and complement) by 2 (num of labels).
                input_batch = {
                    'input_ids': None,
                    'attention_mask': attention_mask,
                    'token_type_ids': None,
                    'inputs_embeds': embeddings,
                    'retain_gradient': False,
                    'extra_info': [{'num_splits': num_splits}, {'num_splits': num_splits}],
                }

                out = self.model(**input_batch)
                # Why [:, 1] -> It is true setup of 1-d binary classification
                prob_out = probability_func(out[0])[:, 1]
                prediction = prob_out * 0.9999 + 5e-6
                all_predictions.append(prediction)

                # Because of the information provided above, we will abandon this masked_idx. As of now we have 2 indexes
                # 0 for masked prediction and 1 for complement masked prediction
                masked_predictions = prediction[0]
                complement_masked_predictions = prediction[1]
                # 10.5 computing loss values for both

                masked_loss = self.compute_masked_loss(masked_predictions, label)
                complement_masked_loss = self.compute_masked_loss(1 - complement_masked_predictions, label)

                classification_loss += masked_loss.sum() + complement_masked_loss.sum()
                # 10.6 considering weights and sigma values in the loss computation to obtain the final loss
                weights_loss = self.weights_loss_factor * torch.square(mask_means := torch.cat(
                    [torch.sum(individual_masks[j], dim=0, keepdim=True) / num_parameters[j] for j in
                     range(len(sets))], dim=0))

                sigma_loss = self.sigma_loss_factor * torch.cat(
                    [torch.sum(-torch.log(individual_sigmas[j]), dim=0, keepdim=True) / num_parameters[j]
                     for j in range(len(sets))], dim=0)


                loss = classification_loss + (sigma_loss + weights_loss).sum()
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

                # 10.7 Trying to check every 50 steps if it is worth to save it
                if optimization_step > 0 and optimization_step % 50 == 0:
                    progress_bar.set_description(
                        f'{description} => {optimization_step}/{self.num_optimization_steps}')
                    mask_means_np = mask_means.detach().cpu().numpy()
                    for each_split in range(len(sets)):

                        diff = abs(last_mask_means[each_split] - mask_means_np[each_split])
                        # 10.8. conditions to show it is worth to save and to collect the final weights
                        if (diff < 1 / 200 or (diff < 1/80 and mask_means_np[each_split] < 0.2)) and (
                            optimization_step >= 199 or mask_means_np[each_split] < 0.3) and sets[each_split] not in final_weights and mask_means_np[each_split] < 0.45:
                            final_weights[sets[each_split]] = individual_masks[each_split]
                            progress_bar.set_description(
                                f'{description} => {sets[each_split]} from {len(sets)} was saved at the optim. step of {optimization_step}')
                        progress_bar.set_description(
                            f'{description} => As of {optimization_step} {len(final_weights)} / {len(sets)}chunks were saved.'
                        )
                    last_mask_means = mask_means_np
                    if len([s for s in sets if s in final_weights]) == len(sets):
                        break
            for j in range(len(sets)):
                if sets[j] not in final_weights:
                    final_weights[sets[j]] = individual_masks[j]
            prev_split_end_prev_set = prev_split_end

        # 11: Combine and compute the resulting weights after the optimization is done
        result_weight = self.process_weights(final_weights, words)

        # NOTE: if access to gt is True, then label is ground truth, otherwise it is prediction
        data = {'weight': result_weight, 'words': words, 'label': label}
        with open(filepath, 'wb') as datapoint_file:
            pickle.dump(data, datapoint_file)


    def process_weights(self, final_weights_data: list, words: list) -> np.ndarray:
        """
        Method combines and computes the resulting weights after the optimization to provide weights for tokens in
        input text
        :param final_weights_data: list of weights for splits
        :param words: list of words
        :return: numpy array of weights for whole document
        """
        result_weights = final_weights_data[0]

        for j in range(1, len(final_weights_data)):
            transition_length = len([w for w in words if j in w['splits'] and j - 1 in w['splits']])
            transition_weight = (torch.arange(start=0, end=transition_length, step=1) / transition_length).to(
                self.configuration.device)
            tensor_val = final_weights_data[j]
            result_weights[-transition_length:] = result_weights[-transition_length:] * (
                    1 - transition_weight) + tensor_val[:transition_length] * transition_weight
            result_weights = torch.cat([result_weights, tensor_val[transition_length:]])

        if result_weights.shape[0] != len(words):
            raise Exception()
        result_weights = result_weights.detach().cpu().numpy()
        return result_weights

    def process_main(self, split='test', set_priority=False) -> None:
        """
        Main process function of the extractor object
        :param split: string to specify which dataset to be used
        :return: None
        """
        gt_info = 'gt' if self.parameters['access_to_gt'] else 'pred'

        self.significance_path = os.path.join(self.configuration.specific_experiment_path_marc, f'result_weights_{gt_info}')
        self.dataset.set_split(split)
        self.idx_name = 'itemid'
        self.rationale_creation_process(set_priority=set_priority)
