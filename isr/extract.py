import os
import sys
from pyarrow.types import is_dictionary

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configuration')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments-marc')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments-echr')))

from torch import softmax
from torch.compiler import assume_constant_result
from tqdm import tqdm
import pickle
import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
from captum.attr import DeepLift
from sklearn.metrics import f1_score, recall_score, precision_score
import math
import copy
from register import RegisterScores

class ExtractionISR:
    def __init__(self, parameters_dict, config_obj, dataset_obj, dataloader_obj, trainer_obj):
        self.parameters = parameters_dict
        self.config = config_obj
        self.dataset = dataset_obj
        self.dataloader = dataloader_obj
        self.trainer = trainer_obj
        self.model = None
        self.limit_ids = list()
        self.register_obj = self.set_register_object()
        self.significance_path = None

    def set_register_object(self):
        register_obj = RegisterScores(
            parameters_dict=self.parameters,
            config_obj=self.config,
            dataset_obj=self.dataset,
            dataloader_obj=self.dataloader,
            trainer_obj=self.trainer
        )
        return register_obj

    def check_dir(self, directory: str) -> None:
        """
        Method checks if the directory exists
        :param directory: path to the directory to be checked
        :return: None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)


    def data_prep_per_creation(self, words, num_splits, split_word_counts, max_split_length):
        split_word_tokens_counts = [[-1] for _ in range(num_splits)]
        split_words = [list() for _ in range(num_splits)]

        # When the length of the text is bigger than 512, authors split the text into splits.
        # Each word can be included with several splits (overlapping for contextual information). Thus code walks
        # through all words and appends num tokens to the corresponding splits of the split_word_tokens_counts list.
        # split_words is a list, of lists, where idx of the lists are representative of the split. Those lists include
        # words that are included by the list.

        for word in words:
            for split in word['splits']:
                split_word_tokens_counts[split].append(word['num_tokens'])  # number of tokens per word to the corresponding split list
                split_words[split].append(word['word'])  # words to the corresponding split list
        num_pads_per_split = list()

        for idx in range(num_splits):
            diff = max_split_length - split_word_counts[idx]  # there might be sum splits that has less elements than max
            # NOTICE: Difference is computed based on number of words (not tokens)
            if diff > 0:
                split_word_tokens_counts[idx] += [-1] * diff  # -1 for special tokens [PAD], [CLS] -> [PAD] here
                split_words[idx] += ['[PAD]'] * diff
            num_pads_per_split.append(diff)

        all_word_tokens_counts = list()

        for i in range(num_splits):

            split_word_tokens_counts[i].append(-1)
            all_word_tokens_counts = all_word_tokens_counts + split_word_tokens_counts[i]

        return split_words, all_word_tokens_counts, split_word_tokens_counts, num_pads_per_split


    def postprocess_scores(self, num_splits, split_word_tokens_counts, combinations, num_pads_per_split):

        scores = dict()
        for split_idx in range(num_splits):

            mask = combinations['variable rationale mask'][split_idx, :]
            weights = combinations['scores'][split_idx, :]
            # each split starts with a special token and ends with one, then padding tokens are added in case necessary.
            # That is why these elements must be removed from each split, since they cause mismatch with num of words in
            # the sequence.
            # split word tokens counts is a list of list, which carry information about how many tokens were created by
            # the specific word. For instance orphanage is tokenized into 2 splits: orphan + ##age. Thus, sum of the
            # elements of the list that corresponds to a split must sum up to the number of elements in the batch - 3
            # special tokens (remember we eliminate them here, but not from batch).
            word_token_counts = split_word_tokens_counts[split_idx][1: -(1 + num_pads_per_split[split_idx])]
            # same applies for attention weights. These are weights were collected from the model, so they are in the
            # same shape of the input (in terms of number of elements). Thus, we eliminate weights that correspond to
            # those special tokens
            split_mask = mask[1: -(1 + num_pads_per_split[split_idx])]
            split_weight = weights[1: -(1 + num_pads_per_split[split_idx])]
            current_idx = 0
            split_masks_combined = list()
            split_weights_combined = list()
            for word_idx, tok_count in enumerate(word_token_counts):
                # since we come here by eliminating special tokens, we don't check if non-one element is -1 or not.
                # if it goes in to this condition, then a word was tokenized into more than 1 token.
                if tok_count != 1:
                    current_word_masks = list()
                    current_word_weights = list()
                    for count in range(tok_count):
                        current_word_masks.append(split_mask[current_idx + count].item())
                        current_word_weights.append(split_weight[current_idx + count].item())
                    # then we take the average of those sub-tokens of the word
                    current_mask = torch.tensor(1) if torch.mean(torch.FloatTensor(current_word_masks)) > 0.5 else torch.tensor(0)
                    current_weight = torch.mean(torch.FloatTensor(current_word_weights))
                else:
                    # else we just take the weight itself :)
                    current_mask = split_mask[current_idx].item()
                    current_weight = split_weight[current_idx].item()
                split_masks_combined.append(current_mask)
                split_weights_combined.append(current_weight)
                # then keep track of it, because number of elements in weight and split are not same
                current_idx += tok_count
            scores[split_idx] = {
                'masks': torch.Tensor(split_masks_combined).to(self.config.device),
                'weights': torch.Tensor(split_weights_combined).to(self.config.device)
            }
        return scores

    def get_rationale_metadata(self, scores_folder: str, result_folder: str) -> None:
        """
        Method gets rationale metadata for all documents, which will be used to create flexible rationales
        :param scores_folder: path to scores folder
        :param result_folder: path to experimental folder, where rationales are saved
        :return:
        """
        meta_data_path = os.path.join(result_folder, 'flexible', f'meta_data_min_{self.parameters["min_length_rationales_isr"]}')
        self.check_dir(meta_data_path)
        metadata_scores = self.register_obj.collect_significance_scores(scores_folder)
        progress_bar = tqdm(
            iterable=enumerate(self.dataset),
            desc=f"{self.parameters['process_name']}: Meta data generation ",
            total=self.dataset.__len__(),
            position=0,
            leave=True,
            file=sys.stdout
        )
        for idx, datapoint in progress_bar:
            text = datapoint['text']
            sample = datapoint['instance']
            if self.parameters['limit_eval']:
                if sample['itemid'] not in self.limit_ids:
                    continue
            file_path = os.path.join(meta_data_path,f"sample_{sample['itemid']}_metadata.pickle")
            if not os.path.exists(file_path):
                progress_bar.set_description(
                    f"{self.parameters['process_name']}: Metadata for {sample['itemid']} "
                    f"does not exist, we create it ..."
                )
                # 1. collect all required information using text, which are tokenized sample and number of splits
                sample_tokenized, num_splits = self.register_obj.collect_req_info(text)
                # 2. create input batch sending those values to the device and update it
                input_batch = {k: v.to(self.config.device) for k, v in sample_tokenized.items()}
                input_batch.update({'inputs_embeds': None, 'retain_gradient': True, 'extra_info': [{'num_splits': num_splits}]})
                # 3. compute the original prediction with the row data, where the each element in the batch is chunk of
                #    the chosen sample
                original_prediction, _ = self.model(**input_batch)
                output = torch.softmax(original_prediction, dim=-1)
                avg_over_chunks = torch.mean(output, dim=0) # FLAG: Fix it for classification
                # 4. get the label using average logits over the batch
                prediction = torch.argmax(avg_over_chunks, dim=-1)
                original_prediction.max(-1)[0].sum().backward(retain_graph=True)
                # 5. initiate a dictionary with all information we already have
                current_dict = {
                    'itemid': sample['itemid'],
                    'original_prediction': original_prediction.detach().cpu().numpy(), # logits - not probability
                    'thresholder': 'contigious',
                    'divergence metric': 'jsd',
                    'prediction': prediction.detach().cpu().numpy(), # label
                }
                progress_bar.set_description(f"{self.parameters['process_name']}: {sample['itemid']} => Data initiated")

                original_sentences_tensor = input_batch["input_ids"].clone()
                only_query_mask = torch.zeros_like(input_batch["input_ids"]).long()
                # 6. mask all input, we will need it
                input_batch["input_ids"] = only_query_mask
                for feat_name in ["lime", "random", "attention", "gradients", "ig", "scaled_attention", "deeplift"]:

                    feat_score = metadata_scores[sample['itemid']][feat_name]
                    # it works well until here - It is adapted as it is needed - Almost nothing changed
                    # 7. compute rationale length for each input score
                    current_dict[feat_name] = self.rationale_length_computer_(
                        inputs=input_batch,
                        scores=feat_score,
                        y_original=original_prediction,
                        original_sents=original_sentences_tensor,
                        rationale_setup_ratio=0.05
                    )
                    progress_bar.set_description(
                        f"{self.parameters['process_name']}:  {sample['itemid']} => {feat_name} rationales created"
                    )

                # 8. That is the flexible rationale element in our dictionary, will be combination of all best options
                #    PER CHUNK
                current_dict["var-len_var-feat"] = [dict()] * original_sentences_tensor.size(0)

                # 8.5. Strategy changed from finding the best scenario for the chunk to the document
                #       Thus, we will run through the feature names, and get the best divergence. According to that div
                #       we will choose the technique per document
                initial_variable_divergence = float('-inf')
                chosen_one = str()
                all_divs = [initial_variable_divergence]
                for feature_scoring_tech in {"attention", "scaled_attention", "gradients", "ig", "lime", "deeplift"}:
                    variable_divergence = current_dict[feature_scoring_tech]['variable-length divergence']
                    if variable_divergence > initial_variable_divergence:
                        chosen_one = feature_scoring_tech
                        initial_variable_divergence = variable_divergence
                    all_divs.append(variable_divergence)

                current_dict['var-len_var-feat'] = {
                    k: v for k, v in current_dict[chosen_one].items()
                }
                current_dict['var-len_var-feat'].update({'feature attribution name': chosen_one})

                with open(file_path, 'wb') as meta_data:
                    pickle.dump(current_dict, meta_data)
                progress_bar.set_description(
                    f"{self.parameters['process_name']}:  {sample['itemid']} => DONE!"
                )
            progress_bar.update(1)

    @staticmethod
    def get_split_info(scoring_data: dict, split_idx: int) -> dict:
        """
        Method helps us to collect all required information for the specific split from the metadata for specific
        scoring information
        :param scoring_data: Scoring data, that includes all combinations for specific feature scoring technique
        :param split_idx: split index that we are interested in
        :return: dictionary of the split we are interested in
        """
        relevant_data = dict()
        for k, v in scoring_data.items():
            if k == 'variable rationale mask':
                relevant_data[k] = v[split_idx, :]
                continue
            relevant_data[k] = v[split_idx]

        return relevant_data

    @staticmethod
    def kl_div_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Method helps us to calculate the KL divergence between two distributions
        :param p: probability distribution
        :param q: probability distribution
        :return: KL divergence value
        """
        # adding 1e-10 for 0 to avoid "inf"
        log_p = torch.log(p + 1e-10)
        log_q = torch.log(q + 1e-10)
        kld = p * (log_p - log_q.float())

        return kld.sum(-1)

    def jsd(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Method helps us to calculate the JSD between two distributions
        :param p: probability distribution
        :param q: probability distribution
        :return: JSD value
        """
        mean = 0.5 * (p + q)
        jsd_val = 0.5 * (self.kl_div_loss(p, mean) + self.kl_div_loss(q, mean))

        return jsd_val

    def get_batch_lengths_info(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Method helps us to get length info, which is basically the special tokens are located, so we can have exact
        position of where the sequence ends
        :param input_ids: tensor for the input indexes, which can be used to detect special tokens
        :return: tensor of those indexes, for all chunks in the document
        """
        batch_lengths = list()
        for split_idx in range(input_ids.shape[0]):
            split = input_ids[split_idx, :]
            pad_token = torch.where(split == self.dataloader.tokenizer.pad_token_id)
            special_token_info = pad_token[0][0].item() if pad_token[0].tolist() else \
            torch.where(split == self.dataloader.tokenizer.sep_token_id)[0].item()
            batch_lengths.append(special_token_info)
        return torch.LongTensor(batch_lengths)

    def rationale_length_computer_(self, inputs, scores, y_original,
            original_sents, rationale_setup_ratio):
        """
        Method collects the best length for each chunks for the given significance scoring technique
        :param inputs:
        :param scores:
        :param y_original:
        :param original_sents:
        :param rationale_setup_ratio:
        :return:
        """
        inputs['retain_gradient'] = False
        # 1. Collect original length of the sequencese
        inputs['lengths'] = self.get_batch_lengths_info(original_sents) # What are the original text size in terms of tokens (when special tokens are discarded) - list of chunks

        # 2. set the rationale setup ratio (upper limit for the rationale can be)
        if original_sents.shape[-1] < 500:
            rationale_setup_ratio = 0.2

        # 2.1 we remove 2 tokens from all chunks' original text (i.e., w/o special tokens) length;
        # 2.2 we take the mean, so we have the average length that represents all the document (kind of) length;
        # 2.3. We use it to determine the maximum length for the rationale
        tokens = rationale_setup_ratio * (inputs["lengths"] - 2).float().mean()
        tokens = int(max(1, torch.round(tokens)))
        # 3. set the lower bound for the rationale length: if it is less than our upper limit, then we have only one
        #    choice to go -> minimum value
        init_point = min(self.parameters['min_length_rationales_isr'], tokens)
        # 4. set the tensor to collect all required information for the specific length
        # there are tokens - init_point + 1 cases that can be collected per chunk, where tokens is the max limit, init_point is the minimum;
        # Thus, when they are 12 and 5, then there are 8 such scenarios per chunk - seems like it :)
        collector_new = torch.zeros([tokens - init_point + 1, original_sents.size(0)])
        grange = range(init_point, tokens + 1)
        predictions_for_lengths = list()

        rationale_length_tracking_list = list()
        divergence_tracking_tensor = torch.zeros([tokens - init_point + 1, 1])
        # 5. run through all possible lengths in the chosen range for this specific document and its chunks
        with (torch.no_grad()):
            for j, _tok in enumerate(grange):
                # 5.1. make sure that we don't pass the upper limit (just in case)
                if _tok > tokens: _tok = tokens
                # 5.2. create rationale mask

                scores_tensor = torch.Tensor(scores)

                rationale_mask = self.create_rationale_mask_(
                    importance_scores=scores_tensor,
                    no_of_masked_tokens=np.array([_tok] * scores_tensor.size(0)),
                )
                # 5.3. create an input with masked out information
                # if rationale then it will be set 0 otherwise it will be own input_ids, since we measure difference when whe mask the rationales out?
                inputs["input_ids"] = (rationale_mask == 0).long() * original_sents
                # 5.4. get the prediction with masked input and collect the label
                yhat, _ = self.model(**inputs)
                predictions_for_lengths.append(yhat.argmax(-1).detach().cpu().item())

                # 5.5 compute the divergence between the logits of the original prediction and prediction wrt masked
                full_div = self.jsd(
                    torch.softmax(y_original.detach().cpu(), dim=-1),
                    torch.softmax(yhat.detach().cpu(), dim=-1)
                )

                divergence_tracking_tensor[j] = full_div.detach().cpu()
                collector_new[j] = full_div.detach().cpu()

                # 5.6. collect also the token information
                rationale_length_tracking_list.append(_tok)

        # 6. get the length's index that causes the max divergence

        maximum_divergence, length_index_max_divergence = divergence_tracking_tensor.max(0)
        chosen_rationale_length = rationale_length_tracking_list[length_index_max_divergence.item()]


        # 7. initialize the dictionary where we will collect relevant information

        sample_dict = {
            "variable rationale length": chosen_rationale_length,
            "variable rationale ratio": list(),
            "variable rationale mask": torch.zeros_like(original_sents).long(),
            f"variable-length divergence": maximum_divergence,
            "running predictions": predictions_for_lengths[length_index_max_divergence.item()],
            'scores': scores
        }
        # 8. collect all relevant information for every chunk
        for _i_ in range(original_sents.shape[0]):
            full_text_length = inputs["lengths"][_i_]
            rationale_ratio = chosen_rationale_length / (full_text_length.float().detach().cpu().item() - 1)
            sample_dict["variable rationale ratio"].append(rationale_ratio)

            rationale_mask = (
                    self.mask_contiguous(
                        original_sents[_i_], scores_tensor[_i_], chosen_rationale_length
                    ) == 0
            ).long()
            sample_dict["variable rationale mask"][_i_, :] = rationale_mask
        sample_dict['variable rationale mask'] = sample_dict['variable rationale mask'].detach().cpu().numpy()

        return sample_dict

    def create_rationale_mask_(self, importance_scores: torch.Tensor, no_of_masked_tokens: np.ndarray) -> torch.Tensor:
        """
        Method helps to create the rationale_mask
        :param importance_scores: Tensor for the significance scores
        :param no_of_masked_tokens: number of masked tokens (length of rationale)
        :return:
        """
        rationale_mask = []

        for _i_ in range(importance_scores.size(0)):

            score = importance_scores[_i_]
            tokens_to_mask = int(no_of_masked_tokens[_i_])

            ## in our scenario, topk does not make sense, since we deal with legal documents
            top_rationales, mean_per_split = self.contiguous_indexes(
                importance_scores=score,
                tokens_to_mask=tokens_to_mask
            )

            rationales = self.combine_sequential_ngrams(top_rationales, tokens_to_mask)

            rationale_indices = [rat['tokens'] for rat in rationales]

            rats = torch.cat(rationale_indices, dim=0).to(self.config.device)
            rationale_indices = torch.unique(rats).long().to(self.config.device)

            ## 1 represents the rationale
            ## 0 represent the other tokens
            mask = torch.zeros(score.shape).to(self.config.device)
            mask = mask.scatter_(-1, rationale_indices.to(self.config.device), 1).long()

            rationale_mask.append(mask)
        # scores can be collected using mask values - 1 are rationales
        rationale_mask = torch.stack(rationale_mask).to(self.config.device)
        return rationale_mask

    @staticmethod
    def contiguous_indexes(importance_scores: torch.Tensor, tokens_to_mask: int) -> tuple:
        """
        Method helps to get the best contiguous indices for rationales by selecting the ones with higher score than mean
        :param importance_scores: significance score for the specific chunk
        :param tokens_to_mask: number of tokens to mask
        :return: list of the best contiguous rationales, and the threshold
        """

        # we take all sliding windows with the given rationale length: for instance [0: 5], [1:6], [2:7], ..., [507: 512]
        # in the given example, tokens_to_mask is 5, and importance scores is a tensor of 512 elements
        ngram = torch.stack(
            [importance_scores[i:i + tokens_to_mask] for i in range(len(importance_scores) - tokens_to_mask + 1)])
        # Then we compute the mean value for a
        threshold = ngram.mean(-1).quantile(0.95)
        indxs = [torch.arange(i, i + tokens_to_mask) for i in range(len(importance_scores) - tokens_to_mask + 1)]

        scores = ngram.mean(-1)

        tops = [
            {
                'tokens': idx_list, 'cont_score': scores[j].item(), 'scores': ngram[j]
            } for j, idx_list in enumerate(indxs) if scores[j].item() >= threshold
        ]

        return tops, threshold

    def combine_sequential_ngrams(self, list_of_rationale_candidates, ngram_size):

        informative_ngram_list = list()
        for token_info in list_of_rationale_candidates:
            info_dict = {
                'tokens': token_info['tokens'].tolist(),
                'cont_score': token_info['cont_score'],
                'scores': token_info['scores'].tolist(),
            }
            informative_ngram_list.append(info_dict)


        result = sorted(informative_ngram_list, key=lambda x: x['tokens'][0], reverse=False)
        merged_rationales = list()
        current_ngram_dict = result[0].copy()

        for ngram_info in result[1:]:

            if ngram_info['tokens'][0] <= current_ngram_dict['tokens'][-1] + 1:
                non_overlapping_scores = [(idx, score) for idx, score in zip(ngram_info['tokens'], ngram_info['scores']) if idx not in current_ngram_dict['tokens']]

                for each in non_overlapping_scores:
                    current_ngram_dict['tokens'].append(each[0])
                    current_ngram_dict['scores'].append(each[1])

            else:
                result = {k: torch.Tensor(v) if k!= 'cont_score' else v for k, v in current_ngram_dict.items()}
                result['cont_score'] = result['scores'].mean()
                merged_rationales.append(result)
                current_ngram_dict = ngram_info.copy()

        result = {k: torch.Tensor(v) if k != 'cont_score' else v for k, v in current_ngram_dict.items()}
        result['cont_score'] = result['scores'].mean()
        merged_rationales.append(result)

        return merged_rationales


    def mask_contiguous(self, sentences: torch.Tensor, scores: torch.Tensor, length_to_mask: int) -> torch.Tensor:
        """
        Method creates contiguous rationales mask
        :param sentences: original input ids
        :param scores: significance scores
        :param length_to_mask: chosen length to mask
        :return: masked input
        """

        top_rationales, mean_per_split = self.contiguous_indexes(scores, length_to_mask)
        rationales = self.combine_sequential_ngrams(top_rationales, length_to_mask)

        qualified_scores_per_split = [rat['tokens'] for rat in rationales]

        rats = torch.cat(qualified_scores_per_split, dim=0).to(self.config.device)
        rationale_indices = torch.unique(rats)
        mask = torch.ones(sentences.shape).to(self.config.device)
        mask = mask.scatter_(-1, rationale_indices.long().to(self.config.device), 0)

        return sentences * mask.long()

    def main_textual_out(self, result_folder: str) -> None:
        """
        Method creates textual output for all techniques in ISR, and flexible one;
        :param result_folder: experimental folder to save output
        :return:
        """

        meta_data_path = os.path.join(result_folder,'flexible', f'meta_data_min_{self.parameters["min_length_rationales_isr"]}')
        feature_name_list = ["lime", "random", "attention", "gradients", "ig", "scaled_attention", "deeplift"]
        for feature_name in feature_name_list:

            sub_result_folder = os.path.join(result_folder, feature_name)
            self.check_dir(sub_result_folder)
            meta_sub_folder = os.path.join(sub_result_folder, f'meta_data_min_{self.parameters["min_length_rationales_isr"]}')
            self.check_dir(meta_sub_folder)

        progress_bar = tqdm(
            os.listdir(meta_data_path),
            total=len(os.listdir(meta_data_path)),
            desc='Generating metas for ',
            leave=False
        )

        attribution_techs = ["lime", "random", "attention", "gradients", "ig", "scaled_attention", "deeplift"]
        for filename in progress_bar:
            fpath = os.path.join(meta_data_path, filename)

            with open(fpath, 'rb') as meta_f:
                meta_data = pickle.load(meta_f)

            for feature_name in attribution_techs:
                meta_folder = os.path.join(result_folder, feature_name, f'meta_data_min_{self.parameters["min_length_rationales_isr"]}')
                new_dir = os.path.join(meta_folder, filename)
                if not os.path.exists(new_dir):
                    progress_bar.set_description(f'Generating metas for {filename}: feature {feature_name}')

                    focus_data = meta_data[feature_name]
                    dict_data = {k: v for k, v in meta_data.items() if k not in attribution_techs}
                    dict_data['var-len_var-feat'] = focus_data.copy()
                    dict_data['var-len_var-feat'].update({'feature attribution name': feature_name})

                    with open(new_dir, 'wb') as meta_f:
                        pickle.dump(dict_data, meta_f)
                else:
                    progress_bar.set_description(f'Generated already metas for {filename}: feature {feature_name}')

        for feature_name in ["flexible", "lime", "random", "attention", "gradients", "ig", "scaled_attention", "deeplift"]:
            print('generating textual output: feature ', feature_name)

            score_path = os.path.join(result_folder, feature_name)
            self.generate_textual_output(score_path)

    def generate_textual_output(self, result_folder: str) -> None:
        """
        Method generates textual output for specific technique that has scores in the given path
        :param result_folder: output folder for the specific technique
        :return: None
        """

        textual_output_path = os.path.join(result_folder, f'rationale_masks_min_{self.parameters["min_length_rationales_isr"]}')
        self.check_dir(textual_output_path)
        progress_bar = tqdm(
            iterable=enumerate(self.dataset),
            desc=f"{self.parameters['process_name']}: Generating textual output ",
            total=self.dataset.__len__(),
            position=0,
            leave=True,
            file=sys.stdout
        )

        meta_data_path = os.path.join(result_folder, f'meta_data_min_{self.parameters["min_length_rationales_isr"]}')
        for idx, datapoint in progress_bar:
            text = datapoint['text']
            sample = datapoint['instance']
            meta_path = os.path.join(meta_data_path, f"sample_{sample['itemid']}_metadata.pickle")
            if not os.path.exists(meta_path):
                continue
            file_path = os.path.join(textual_output_path, f"sample_{sample['itemid']}.pickle")


            if not os.path.exists(file_path):

                with open(meta_path, 'rb') as meta_f:
                    meta_data = pickle.load(meta_f)

                best_choice = meta_data['var-len_var-feat']

                progress_bar.set_description(
                    f"{self.parameters['process_name']}: Extracting textual output {sample['itemid']}")

                words, num_splits, split_word_counts = self.dataloader.split_sample_legal_bert(text)
                max_split_length = max(split_word_counts.values())
                split_words, all_word_tokens_counts, split_word_tokens_counts, num_pads_per_split = self.data_prep_per_creation(
                    words, num_splits, split_word_counts, max_split_length)

                masked_text, text_weights = self.transition_rationales(
                    best_choice,
                    words,
                    num_splits,
                    num_pads_per_split,
                    split_word_tokens_counts
                )

                data = {
                    'words': words, 'label': meta_data['prediction'], 'weight': text_weights, 'mask': masked_text,
                }

                with open(file_path, 'wb') as masked_file:
                    pickle.dump(data, masked_file)

    def transition_rationales(self, weights, words, num_splits, pad_splits, word_tokens) -> tuple:
        """
        Method creates transition rationales mask for overlapping chunks
        :param weights: importance scores for chunks [the best ones]
        :param words:
        :param num_splits: number of chunks
        :param pad_splits: number of paddigns per split
        :param word_tokens:
        :return: Tuple of binary mask and words with updated signficance scores
        """
        word_weights = self.postprocess_scores(num_splits, word_tokens, weights, pad_splits)

        result_mask = word_weights[0]['masks']
        result_weight = word_weights[0]['weights']

        for idx in range(1, num_splits):
            # tokens that occur in both splits
            transition_length = len([w for w in words if idx in w['splits'] and idx - 1 in w['splits']])
            # linear interpolation weight in the length of this difference
            transition_weight = (torch.arange(start=0, end=transition_length, step=1) / transition_length).to(self.config.device)
            tensor_mask = word_weights[idx]['masks']
            tensor_weight = word_weights[idx]['weights']
            result_mask[-transition_length:] = result_mask[-transition_length:] * (1 - transition_weight) + tensor_mask[
                                                                                                                :transition_length] * transition_weight
            result_weight[-transition_length:] = result_weight[-transition_length:] * (1 - transition_weight) + tensor_weight[
                                                                                                            :transition_length] * transition_weight
            result_mask = torch.cat([result_mask, tensor_mask[transition_length:]])
            result_weight = torch.cat([result_weight, tensor_weight[transition_length:]])

        result = (result_mask > 0.5).long()
        result = result.detach().cpu().numpy()
        result_weights = result_weight.detach().cpu().numpy()
        if result_mask.shape[0] != len(words):
            raise Exception()
        if result_weight.shape[0] != len(words):
            raise Exception()
        return result, result_weights

    def evaluate_faithfulness_per_sample(self, label, mask, words, splits, split_counts, probability_func, max_length):

        mask_randomness = np.random.randn(max_length) * 1e-5

        mask += mask_randomness[:len(mask)]
        sorted_scores = np.sort(mask)[::-1]
        sufficiency = 0
        comprehensiveness = 0
        predictions = dict()
        for test in ["sufficiency", "comprehensiveness"]:
            current_values = list()
            for percentage in range(0, 105, 5):
                words_copy = copy.deepcopy(words)
                if percentage == 0:
                    score_threshold = sorted_scores[0] + 0.001
                elif percentage == 100:
                    score_threshold = sorted_scores[-1] - 0.001
                else:
                    score_threshold = sorted_scores[int(np.round(len(sorted_scores) * (percentage / 100)))]

                # Remove masked words and create new input text
                for j in range(len(mask)):
                    if (mask[j] <= score_threshold and test == "sufficiency") or (
                            mask[j] > score_threshold and test == "comprehensiveness"):
                        words_copy[j]["tokens"] = ["[PAD]"] * len(words_copy[j]["tokens"])
                input_texts = [" ".join(
                    ["".join(words_copy[n]["tokens"]) for n in range(len(mask)) if s in words_copy[n]["splits"]])
                    for s in range(splits)]

                # Add pad tokens to make all splits have same length
                max_length = max(split_counts.values())
                for j in range(len(input_texts)):
                    input_texts[j] += " [PAD]" * (max_length - split_counts[j])

                input_texts_tokenized = self.dataloader.tokenizer(input_texts, return_tensors='pt',
                                                                  truncation=False).to("cuda")

                input_batch = {
                    'input_ids': input_texts_tokenized['input_ids'],
                    'attention_mask': input_texts_tokenized['attention_mask'],
                    'token_type_ids': input_texts_tokenized['token_type_ids'],
                    'inputs_embeds': None,

                }
                if self.parameters['filre_model']:
                    input_batch['retain_gradient'] = False
                # out = self.model(**input_batch) if input_batch['input_ids'].shape[]
                out = self.model(**input_batch)[0]
                torch.cuda.empty_cache()

                prediction = torch.mean(probability_func(out)[:, 1], dim=0) if self.parameters['filre_model'] \
                    else torch.mean(probability_func(out), dim=0)

                current_values.append(prediction.detach().cpu().numpy())
            predictions[test] = 1 if np.mean(current_values[1: -1]) > 0.5 else 0
            current_values = [label * x + (1 - label) * (1 - x) for x in current_values]  # sample[1] -> label

            if test == "sufficiency":
                sufficiency = np.mean([current_values[-1] - c for c in current_values[1:-1]])
            elif test == "comprehensiveness":
                comprehensiveness = np.mean([current_values[0] - c for c in current_values[1:-1]])
        return sufficiency, comprehensiveness, predictions

    def __main__(self, split, set_priority=False):
        self.register_obj.__main__(split, set_priority)
        self.model = self.register_obj.model
        self.limit_ids = self.register_obj.limit_ids
        self.dataset.set_split(split)
        self.significance_path = self.register_obj.significance_path

        scores_path = os.path.join(self.significance_path, 'scores')

        result_path = os.path.join(self.significance_path, 'results')
        if not os.path.exists(scores_path):
            raise NotImplementedError('You should register scores first!')
        self.get_rationale_metadata(scores_path, result_path)
        self.main_textual_out(result_path)


