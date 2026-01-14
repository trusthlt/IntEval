import copy
import pickle
import sys

import numpy as np
import os
from tqdm import tqdm
import torch
from typing import Union
from sklearn.metrics import f1_score, confusion_matrix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configuration')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments-isr-new')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments-marc')))



class QuantitativeEvaluation:
    def __init__(self, extractor_obj, tech_name: str,
                 config_obj, parameters: dict, dataset_obj):
        self.extractor = extractor_obj
        self.model = self.extractor.model
        self.tech = tech_name
        self.limit_ids = self.extractor.register_obj.limit_ids if self.tech == 'isr' else self.extractor.limit_ids
        self.config_obj = config_obj
        self.parameters = parameters
        self.dataset = dataset_obj
        self.experiment_folder = self.set_experiment_folder()

    def set_experiment_folder(self) -> os.path:
        """
        Method sets the experiment folder, for generalizing evaluation process
        :return: path to the experiment folder for specific technique
        """
        exp_path = self.extractor.significance_path
        results_folder = os.path.join(exp_path, 'results')
        if not os.path.exists(results_folder):
            raise NotImplementedError('You should have created the results folder')
        return results_folder

    def create_masked_input(self, original_input: dict, rationale_mask: Union[torch.Tensor, np.ndarray],
                            input_batch: dict, only_query_mask: torch.Tensor, is_sufficiency: bool=True):
        """
        Metho creates a masked input tensor for specific technique (again for generalizing evaluation process)
        :param original_input: if technique is isr, then the original input is original input indexes to be masked
                               if technique is marc, then this is a dictionary that holds number of splits, list of
                                words and list of word counts for splits, because they will be used for masking
        :param rationale_mask: np.array for marc, tensor for isr -> rationale mask values
        :param input_batch: dictionary for input batch
        :param only_query_mask: zero input mask
        :param is_sufficiency: specifies whether the mask is created for sufficiency (True) or comprehensiveness (False)
        :return:
        """

        if self.tech == 'isr':

            mask = rationale_mask if is_sufficiency else (rationale_mask == 0)
            mask[:, 0] = 1

            mask[torch.arange(mask.shape[0]).to(self.config_obj.device), input_batch['lengths']] = 1
            inputs = copy.deepcopy(input_batch)

            inputs['input_ids'] = (mask * original_input['original_sentences']).long().to(self.config_obj.device)

        elif self.tech == 'marc':
            words = original_input['words']
            num_splits = original_input['num_splits']
            split_word_counts = original_input['split_word_counts']
            score_threshold = 0.2
            inputs = copy.deepcopy(input_batch)

            for j in range(len(rationale_mask)):
                if is_sufficiency:
                    if rationale_mask[j] <= score_threshold:  # we remove context but keep only rationales :)
                        words[j]["tokens"] = ["[PAD]"] * len(words[j]["tokens"])
                else:
                    if rationale_mask[j] > score_threshold:  # we remove rationales :)
                        words[j]["tokens"] = ["[PAD]"] * len(words[j]["tokens"])
            input_texts = [" ".join(
                ["".join(words[n]["tokens"]) for n in range(len(rationale_mask)) if s in words[n]["splits"]])
                for s in range(num_splits)]

            max_length = max(split_word_counts.values())
            for j in range(len(input_texts)):
                input_texts[j] += " [PAD]" * (max_length - split_word_counts[j])
            input_texts_tokenized = self.extractor.dataloader.tokenizer(input_texts, return_tensors='pt',
                                                              truncation=False).to("cuda")

            inputs.update({
                "input_ids": input_texts_tokenized["input_ids"],
                "attention_mask": input_texts_tokenized["attention_mask"],
                "token_type_ids": input_texts_tokenized["token_type_ids"],
                "retain_gradient": False,
                "inputs_embeds": None
            })
        else:
            raise NotImplementedError
        return inputs

    @staticmethod
    def compute_sufficiency(full_text_probs: np.ndarray, reduced_probs: np.ndarray) -> np.ndarray:
        """
        Method computes the sufficiency score for the given original and reduced probabilities
        :param full_text_probs: original prediction
        :param reduced_probs: model outcome with masked input
        :return: sufficiency value
        """
        sufficiency = 1 - np.maximum(0, full_text_probs - reduced_probs)
        return sufficiency

    def compute_norm_sufficiency(self, original_input: dict, rationale_mask: torch.Tensor, input_batch: dict,
                                 rows: np.ndarray, full_text_probs: np.ndarray, full_text_class: np.ndarray,
                                 sufficiency_y_zero: np.ndarray, only_query_mask: torch.Tensor=None) -> tuple:
        """
        Method computes the normalized comprehensiveness score. Masked input creation varies with the given technique,
        so "inputs" is created for specific technique (again for generalizing evaluation process)
        :param original_input: if isr: then we have the original sentence indexes as an only element of the dictionary
                               if marc: then we have the dictionary of number of splits, list of words and
                               list of word counts to be used for mask creation
        :param rationale_mask: np.array for marc, tensor for isr -> rationale mask values
        :param input_batch: dictionary for input batch
        :param rows: number of rows in the input batch (number of splits)
        :param full_text_probs: original prediction
        :param full_text_class: class of the original prediction
        :param sufficiency_y_zero: baseline sufficiency with zero input mask
        :param only_query_mask: zero input mask
        :return: normalized sufficiency value and reduced probability when input includes only rationales
        """

        masked_input = self.create_masked_input(
            original_input, rationale_mask, input_batch, only_query_mask, is_sufficiency=True
        )

        masked_yhat, _ = self.model(**masked_input)

        masked_yhat = torch.softmax(masked_yhat.detach().cpu(), dim=-1).numpy()
        # because we have only one outcome, we don't need to use rows as before here (before we were using predictions
        # per chunks
        reduced_probs = masked_yhat[0, full_text_class]

        ## reduced input sufficiency
        sufficiency_y_a = self.compute_sufficiency(full_text_probs, reduced_probs)

        # return suff_y_a
        sufficiency_y_zero -= 1e-4  ## to avoid nan

        norm_suff = np.maximum(0, (sufficiency_y_a - sufficiency_y_zero) / (1 - sufficiency_y_zero))

        norm_suff = np.clip(norm_suff, a_min=0, a_max=1)

        return norm_suff, reduced_probs


    @staticmethod
    def compute_comprehensiveness(full_text_probs: np.ndarray, reduced_probs: np.ndarray) -> np.array:
        """
        Method computes the comprehensiveness score for the given original and reduced probabilities
        :param full_text_probs: original prediction
        :param reduced_probs: model outcome with masked input
        :return: comprehensiveness value
        """
        comprehensiveness = np.maximum(0, full_text_probs - reduced_probs)
        return comprehensiveness

    def compute_norm_comprehensiveness(self, original_input: dict, rationale_mask: torch.Tensor, input_batch: dict,
                                       rows: np.ndarray, full_text_probs: np.ndarray, full_text_class: np.ndarray,
                                       sufficiency_y_zero: np.ndarray, only_query_mask: torch.Tensor=None):
        """
        Method computes the normalized comprehensiveness score. Masked input creation varies with the given technique,
        so "inputs" is created for specific technique (again for generalizing evaluation process)
        :param original_input: if isr: then we have the original sentence indexes as an only element of the dictionary
                               if marc: then we have the dictionary of number of splits, list of words and
                               list of word counts to be used for mask creation
        :param rationale_mask: np.array for marc, tensor for isr -> rationale mask values
        :param input_batch: dictionary for input batch
        :param rows: number of rows in the input batch (number of splits)
        :param full_text_probs: array of predictions of each split
        :param full_text_class: array of classes of the original prediction of each split
        :param sufficiency_y_zero: baseline sufficiency with zero input mask
        :param only_query_mask: zero input mask
        :return: tuple of normalized comprehensiveness value and prediction of the model when input doesn't have
                 rationales in the input
        """
        masked_input = self.create_masked_input(
            original_input, rationale_mask, input_batch, only_query_mask, is_sufficiency=False
        )

        masked_y_hat, _ = self.model(**masked_input)

        masked_y_hat = torch.softmax(masked_y_hat, dim=-1).detach().cpu().numpy()

        reduced_probs = masked_y_hat[0, full_text_class]

        ## reduced input comprehensiveness
        comp_y_a = self.compute_comprehensiveness(full_text_probs, reduced_probs)

        sufficiency_y_zero -= 1e-4  # to avoid nan

        ## 1 - suff_y_0 == comp_y_1
        norm_comp = np.maximum(0, comp_y_a / (1 - sufficiency_y_zero))

        norm_comp = np.clip(norm_comp, a_min=0, a_max=1)

        return norm_comp, reduced_probs

    def evaluate_technique(self) -> None:
        """
        Method is the evaluation manager for the whole generalized process, that can adapt evaluation process to the
        given technique
        :return: None
        """
        self.dataset.set_split('test')
        feature_name_list = [
            "random", "flexible", "lime", "attention", "gradients", "ig", "scaled_attention", "deeplift"
        ] if self.tech == 'isr' else [""]

        for feature_name in feature_name_list:

            score_folder = os.path.join(self.experiment_folder, feature_name)
            if not feature_name:
                feature_name = 'marc'
            if not os.path.exists(score_folder):
                raise NotImplementedError(f"There is not masks for feature name {feature_name}")
            print(f'<<<<<<<<<{feature_name}>>>>>>>>>>')
            self.evaluate_faithfulness(score_folder, feature_name)

    def evaluate_faithfulness(self, result_folder: os.path, feature_name: str) -> None:
        """
        Method evaluates organizes computing and saving process of the evaluation results
        :param result_folder: folder where results are saved
        :param feature_name: feature name to show us what is going on and specifies the result folder for isr's inner
                             methods
        :return: None
        """
        faith_values = dict()
        meta_data_folder = os.path.join(result_folder, f'meta_data_min_{self.parameters["min_length_rationales_isr"]}') \
            if self.tech == 'isr' else os.path.join(result_folder, 'meta_data_marc')
        num_data = self.dataset.__len__()

        extra_info = "_metadata" if self.tech == 'isr' else ""
        faith_values_path = os.path.join(result_folder, 'faithfulness_values.pickle')

        if not os.path.exists(faith_values_path):
            for test in ['sufficiency', 'comprehensiveness']:

                progress_bar = tqdm(
                    iterable=enumerate(self.dataset),
                    desc=f"{feature_name} => {self.parameters['process_name']}: Evaluation process is started",
                    total=num_data,
                    position=0,
                    leave=True,
                )
                for idx, datapoint in progress_bar:
                    text = datapoint['text']
                    label = datapoint['labels']
                    sample = datapoint['instance']

                    if self.parameters['limit_eval']:
                        if sample['itemid'] not in self.limit_ids:
                            continue
                    meta_file = os.path.join(meta_data_folder, f'sample_{sample["itemid"]}{extra_info}.pickle')
                    with open(meta_file, 'rb') as meta_path:
                        rationale_metadata = pickle.load(meta_path)

                    sample_tokenized, split_info = self.extractor.collect_req_info(text) if self.tech == 'marc' else self.extractor.register_obj.collect_req_info(text)
                    num_splits = split_info['num_splits'] if self.tech == 'marc' else split_info

                    sample_tokenized['query_mask'] = sample_tokenized['attention_mask'].clone()
                    input_batch = {k: v.to(self.config_obj.device) for k, v in sample_tokenized.items()}
                    input_batch.update({'inputs_embeds': None, 'retain_gradient': False, 'extra_info':[{'num_splits': num_splits}]})
                    with torch.no_grad():
                        if self.tech == 'marc':
                            y_hat, _ = self.model(**input_batch)
                            torch.cuda.empty_cache()
                            rationale_mask = rationale_metadata['weight']
                            original_sentences = split_info

                        else:
                            original_sentences = {'original_sentences': input_batch["input_ids"].clone()}
                            y_hat = torch.Tensor(rationale_metadata['original_prediction']).to(self.config_obj.device)
                            input_batch['lengths'] = self.extractor.get_batch_lengths_info(original_sentences['original_sentences']) # make sure if it is from register or extractor
                            masks = rationale_metadata['var-len_var-feat']['variable rationale mask']

                            rationale_mask = torch.Tensor(masks).to(self.config_obj.device)
                        original_prediction = torch.softmax(y_hat, dim=-1).detach().cpu().numpy()

                        prediction = np.mean(original_prediction, axis=0).argmax(axis=-1)

                        full_text_probs = original_prediction.max(-1)
                        full_text_class = original_prediction.argmax(-1)

                        rows = np.arange(input_batch['input_ids'].size(0))

                        only_query_mask = torch.zeros_like(input_batch["input_ids"]).long()

                        input_batch["input_ids"] = only_query_mask

                        yhat, _ = self.model(**input_batch)

                        yhat = torch.softmax(yhat, dim=-1).detach().cpu().numpy()

                        reduced_probs = yhat[0, full_text_class]
                        sufficiency_y_zero = self.compute_sufficiency(
                            full_text_probs,
                            reduced_probs
                        )

                        if test == 'sufficiency':
                            sufficiency, reduced_probs_sufficiency = self.compute_norm_sufficiency(
                                original_input=original_sentences, rationale_mask=rationale_mask,
                                input_batch=input_batch, rows=rows, full_text_probs=full_text_probs,
                                full_text_class=full_text_class, sufficiency_y_zero=sufficiency_y_zero,
                                only_query_mask=only_query_mask
                            )
                            sufficiency_val = np.mean(sufficiency)
                            mask_prediction = 1 if np.mean(reduced_probs_sufficiency) > 0.5 else 0


                            faith_values[sample['itemid']] = {
                                'sufficiency': sufficiency_val, 'masked_predictions_sufficiency': mask_prediction,
                                'original_predictions': prediction,
                                'ground_truth': label
                            }
                            torch.cuda.empty_cache()

                        if test == 'comprehensiveness':
                            comprehensiveness, reduced_probs_comprehensiveness = self.compute_norm_comprehensiveness(
                                original_input=original_sentences, rationale_mask=rationale_mask,
                                input_batch=input_batch, rows=rows, full_text_probs=full_text_probs,
                                full_text_class=full_text_class, sufficiency_y_zero=sufficiency_y_zero,
                                only_query_mask=only_query_mask
                            )
                            torch.cuda.empty_cache()
                            mask_prediction = 1 if np.mean(reduced_probs_comprehensiveness) > 0.5 else 0

                            comprehensiveness_val = np.mean(comprehensiveness)
                            faith_values[sample['itemid']]['comprehensiveness'] = comprehensiveness_val
                            faith_values[sample['itemid']]['masked_predictions_comprehensiveness'] = mask_prediction
            with open(faith_values_path, 'wb') as faith_values_file:
                pickle.dump(faith_values, faith_values_file)

        with open(faith_values_path, 'rb') as faith_values_file:
            faith_values = pickle.load(faith_values_file)

        print(f'Average sufficiency: {sum(dicts["sufficiency"] for dicts in faith_values.values()) / num_data: .4f},'
              f'Average comprehensiveness: {sum(dicts["comprehensiveness"] for dicts in faith_values.values()) / num_data: .4f},')

        gts = list()
        masked_sufficiency = list()
        masked_comprehensiveness = list()
        preds = list()
        for sample_results in faith_values.values():
            gts.append(sample_results['ground_truth'])
            masked_sufficiency.append(sample_results['masked_predictions_sufficiency'])
            masked_comprehensiveness.append(sample_results['masked_predictions_comprehensiveness'])
            preds.append(sample_results['original_predictions'])

        f1_original = f1_score(gts, preds, average='macro')
        f1_masked_s = f1_score(gts, masked_sufficiency, average='macro')
        f1_wrt_pred_s = f1_score(preds, masked_sufficiency, average='macro')
        f1_masked_c = f1_score(gts, masked_comprehensiveness, average='macro')
        f1_wrt_pred_c = f1_score(preds, masked_comprehensiveness, average='macro')

        print(f'COMP   original f1: {f1_original:.4f}, masked f1: {f1_masked_c:.4f}, wrt pred: {f1_wrt_pred_c:.4f}')
        print(f'SUFF   original f1: {f1_original:.4f}, masked f1: {f1_masked_s:.4f}, wrt pred: {f1_wrt_pred_s:.4f}')
