import os
import sys

from tqdm import tqdm
import pickle
import torch
import numpy as np
from lime.lime_text import LimeTextExplainer


from predictors import LimePredictor, ShapleyModelWrapper
from captum.attr import DeepLift

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configuration')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments-marc')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments-echr')))

# from setup import Configuration
# from data_processing.dataset import ECHRVio
# from data_processing.model_data import DataLoadECHR
# from trainer import TrainModel


class RegisterScores:
    def __init__(self, parameters_dict: dict, config_obj, dataset_obj,
                 dataloader_obj, trainer_obj) -> None:
        self.parameters = parameters_dict
        self.config = config_obj
        self.dataset = dataset_obj
        self.dataloader = dataloader_obj
        self.trainer = trainer_obj
        self.significance_path = None
        self.model = None
        self.set_model()
        self.limit_ids = list()


    def set_model(self) -> None:
        """
        Method sets up the model, load and make it ready for evaluation
        :return: None
        """
        self.trainer.test_model()
        self.model = self.trainer.model

    def check_dir(self, directory: str) -> None:
        """
        Method checks if the directory exists
        :param directory: path to the directory to be checked
        :return: None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def data_prep_per_creation(self, words: list, num_splits: int, split_word_counts: dict, max_split_length: int) -> tuple:
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

    def compute_importance(self, input_batch: dict) -> dict:
        """
        Method is used to compute and register significance scores for documents with affordable amount of chunks (30)
        :param input_batch: input batch for significance computation
        :return: dictionary for significance scores for given batch
            Note: This batch is chunks of each document. If number of chunks is more than threshold (30) then we create
            several groups of 30 chunks as a batch and feed the model with them and collect. Thus, this input batch
            does not necessarily represent a document but can be some part of it.
        """
        # print(input_batch)
        yhat, attentions = self.model(**input_batch)
        # input('now check')

        yhat.max(-1)[0].sum().backward(retain_graph=True)

        embed_grad = self.model.core_model.model.embeddings.word_embeddings.weight.grad
        # collect all here
        # print('so?')
        # input('ready?')
        # attention_gradients = self.model.weights_or.grad[:, :, 0, :].mean(1)
        # input('yes it bursts here')



        scores_dict = dict()
        g = embed_grad[input_batch["input_ids"].long()]



        em = self.model.core_model.model.embeddings.word_embeddings.weight[input_batch["input_ids"].long()]


        # --------------- Normalised Attention -------------- #

        scores_dict['attention'] = torch.masked_fill(attentions, ~input_batch["query_mask"].bool(), float("-inf"))


        # retrieving attention*attention_grad
        attention_gradients = self.model.weights_or.grad[:, :, 0, :].mean(1)

        attention_gradients = (attentions * attention_gradients)


        scores_dict['scaled_attention'] = torch.masked_fill(attention_gradients, ~input_batch["query_mask"].bool(),
                                                            float("-inf"))

        # --------------- Integrated Gradients ------------- #
        gradients = (g * em).sum(-1).abs() * input_batch["query_mask"].float()
        ig = self.model.integrated_grads(original_grad=g, original_pred=yhat.max(-1),
                                                                      **input_batch)
        # --------------- Normalised Random -------------- #
        normalised_random = torch.randn(attentions.shape).to(self.config.device)

        scores_dict['random'] = torch.masked_fill(normalised_random, ~input_batch["query_mask"].bool(), float("-inf"))

        # --------------- Normalised IG of input -------------- #
        scores_dict['ig'] = torch.masked_fill(ig, ~input_batch["query_mask"].bool(), float("-inf"))

        # --------------- Normalised Gradients of input -------------- #
        scores_dict['gradients'] = torch.masked_fill(gradients, ~input_batch["query_mask"].bool(), float("-inf"))

        return scores_dict

    def compute_importance_large_documents(self, input_batch: dict) -> dict:
        """
        Method is used to compute importance scores for larger documents (i.e., docs with chunks of more than 30)
        :param input_batch: input batch for importance computation
        :return: dictionary with importance scores for all documents
        """
        result_scores = dict()
        num_splits = input_batch['input_ids'].shape[0]

        init = 0
        num_steps = 30
        for split in range(30, num_splits, num_steps):
            if num_splits - split <= num_steps:
                split = num_splits

            sub_input_batch = {
                k: v[init:split, :] if k not in ['inputs_embeds', 'retain_gradient', 'extra_info'] else v
                for k, v in input_batch.items()
            }
            sub_input_batch.update({'extra_info': [{'num_splits': split - init}]})

            scores = self.compute_importance(sub_input_batch)
            for score_name in scores.keys():
                if score_name not in result_scores:
                    result_scores[score_name] = torch.zeros_like(
                        input_batch['input_ids'], dtype=scores[score_name].dtype
                    )
                result_scores[score_name][init: split, :] = scores[score_name]
            init = split

        return result_scores

    def collect_req_info(self, text: str) -> tuple:
        """
        Method is used to collect required information for data processing before the importance computation
        :param text: text input to be processed
        :return: tuple of tokenized sample and number of chunks
        """
        words, num_splits, split_word_counts = self.dataloader.split_sample_legal_bert(text)
        max_split_length = max(split_word_counts.values())
        split_words, all_word_tokens_counts, split_word_tokens_counts, num_pads_per_split = self.data_prep_per_creation(
            words, num_splits, split_word_counts, max_split_length)

        sample_tokenized = self.dataloader.tokenizer(split_words, return_tensors='pt', truncation=False,
                                                     is_split_into_words=True)

        # --------------- ADDING QUERY MASK FOR ISR SETUP --------------------- #
        # the following 2 lines are taken from ISR dataholder's encodeplusplus part, since it will be needed
        sample_tokenized["query_mask"] = sample_tokenized["attention_mask"].clone()
        return sample_tokenized, num_splits

    def extract_importance(self, result_folder: str, set_priority=False) -> None:
        """
        Method is used to extract importance scores for all documents. It simply walks through the dataset and
        computes significance scores for each sample
        :param result_folder: experimental folder to save the importance scores
        :return: None
        """
        importance_scores_path = os.path.join(result_folder, f'importance_scores')
        self.check_dir(importance_scores_path)
        progress_bar = tqdm(
            iterable=enumerate(self.dataset),
            desc=f"{self.parameters['process_name']}: Registering importance scores for ",
            total=self.dataset.__len__(),
            position=0,
            leave=True,
            file=sys.stdout
        )
        priority_list = [
            '001-120512', '001-201325', '001-79415', '001-92582', '001-94578',
            '001-145215', '001-175482', '001-191714', '001-219778', '001-228385'
        ]
        for idx, datapoint in progress_bar:

            text = datapoint['text']
            sample = datapoint['instance']
            if set_priority and sample['itemid'] not in priority_list:
                self.parameters['limit_eval'] = False
                continue

            if self.parameters['limit_eval'] and idx == self.parameters['limit_eval']:
                break
            self.limit_ids.append(sample['itemid'])
            file_path = os.path.join(importance_scores_path, f"sample_{sample['itemid']}_importance.pickle")

            if not os.path.exists(file_path):

                progress_bar.set_description(
                    f"{self.parameters['process_name']}: Registering importance scores for {sample['itemid']}")
                sample_tokenized, num_splits = self.collect_req_info(text)

                input_batch = {k: v.to(self.config.device) for k, v in sample_tokenized.items()}

                input_batch.update(
                    {'inputs_embeds': None, 'retain_gradient': True, 'extra_info': [{'num_splits': num_splits}]}
                )
                result_scores = self.compute_importance_large_documents(input_batch) if num_splits > 30 \
                    else self.compute_importance(input_batch)
                with open(file_path, 'wb') as importance_data:
                    pickle.dump(result_scores, importance_data)
                progress_bar.set_description(
                    f"{self.parameters['process_name']}: Importance scores for {sample['itemid']} was saved successfully!")
            progress_bar.update(1)


    def extract_lime_scores(self, result_folder: str) -> None:
        """
                Method walks through the dataset and computes lime scores for all documents
                :param result_folder: experimental folder to save the lime scores
                :return: None
                """
        progress_bar = tqdm(
            iterable=enumerate(self.dataset),
            desc=f"{self.parameters['process_name']}: Registering lime scores for ",
            total=self.dataset.__len__(),
            position=0,
            leave=True,
            file=sys.stdout
        )

        lime_scores_path = os.path.join(result_folder, f'lime_scores')
        self.check_dir(lime_scores_path)

        lime_predictor = LimePredictor(self.parameters, self.model, self.dataloader, None, self.config)

        explainer = LimeTextExplainer(class_names=list(range(2)), split_expression=" ")

        if not self.limit_ids:
            raise NotImplementedError('Collect significance scores first, which collects limited number of documents')

        for idx, datapoint in progress_bar:

            text = datapoint['text']
            sample = datapoint['instance']
            file_path = os.path.join(lime_scores_path, f"sample_{sample['itemid']}_lime.pickle")
            if not os.path.exists(file_path):

                if self.parameters['limit_eval']:
                    if sample['itemid'] not in self.limit_ids:
                        continue

                progress_bar.set_description(
                    f"{self.parameters['process_name']}: Registering lime scores for {sample['itemid']}")
                sample_tokenized, num_splits = self.collect_req_info(text)

                tokenized_text = self.dataloader.tokenizer.tokenize(text)
                updated_text = ' '.join(tokenized_text)

                exp = explainer.explain_instance(
                    updated_text,
                    lime_predictor.predictor,
                    num_samples=500,
                    num_features=len(set(tokenized_text))
                )

                word_explanations = dict(exp.as_list())

                scores_for_data = list()
                for split_idx in range(num_splits):
                    split = sample_tokenized['input_ids'][split_idx, :]
                    features_for_split = self.dataloader.tokenizer.convert_ids_to_tokens(split)
                    lime_scores_for_split = [word_explanations[feature] if feature in word_explanations else 0. for feature in features_for_split]
                    scores_for_data.append(lime_scores_for_split)
                pickle.dump(torch.Tensor(scores_for_data), open(file_path, 'wb'))

        print('Lime scores are saved successfully!')


    def extract_deeplift_values(self, result_folder: str) -> None:
        progress_bar = tqdm(
            iterable=enumerate(self.dataset),
            desc=f"{self.parameters['process_name']}: Registering shap scores for ",
            total=self.dataset.__len__(),
            position=0,
            leave=True,
            file=sys.stdout
        )

        shap_scores_path = os.path.join(result_folder, f'shap_scores')
        self.check_dir(shap_scores_path)
        explainer = DeepLift(ShapleyModelWrapper(self.model))

        for idx, datapoint in progress_bar:

            text = datapoint['text']
            sample = datapoint['instance']
            if self.parameters['limit_eval']:
                if sample['itemid'] not in self.limit_ids:
                    continue
            file_path = os.path.join(shap_scores_path, f"sample_{sample['itemid']}_shap.pickle")
            if not os.path.exists(file_path):

                sample_tokenized, num_splits = self.collect_req_info(text)
                input_batch = {k: v.to(self.config.device) for k, v in sample_tokenized.items()}

                input_batch.update({'inputs_embeds': None, 'retain_gradient': True, 'extra_info': [{'num_splits': num_splits}]})
                attribution = self.deeplift_compute(explainer, input_batch)

                with open(file_path, 'wb') as shap_path:
                    pickle.dump(torch.Tensor(attribution), shap_path)
                progress_bar.update(1)

        print('Deeplift scores are saved successfully!')

    def deeplift_compute(self, explainer: DeepLift, input_batch: dict) -> torch.Tensor:
        """
        Method computes the deeplift scores for documents has affordable amount of chunks (30)
        :param explainer: explainer object
        :param input_batch: input batch for deeplift computation
        :return: feature attribution for the given input batch
        """
        original_prediction, _ = self.model(**input_batch)
        embeddings = self.model.core_model.model.embeddings.word_embeddings.weight[input_batch['input_ids'].long()]
        attribution = explainer.attribute(
            embeddings.requires_grad_(True),
            target=original_prediction.argmax(-1),
        )

        attribution = attribution.sum(-1)

        attribution = torch.masked_fill(
            attribution,
            (input_batch['query_mask'] == 0).bool(),
            float("-inf")
        )

        return attribution

    def deeplift_compute_large_document(self, explainer: DeepLift, input_batch: dict) -> torch.Tensor:
        """
        Method computes the deeplift scores for documents has larger amount of chunks (>30)
        :param explainer: explainer object
        :param input_batch: input batch for deeplift computation
        :return: feature attribution for the given input batch
        """
        init = 0
        num_steps = 30
        num_splits = input_batch['input_ids'].shape[0]
        result_attribution = torch.zeros_like(input_batch['input_ids'], dtype=torch.float32)
        for split in range(30, num_splits, num_steps):
            if num_splits - split <= num_steps:
                split = num_splits
            sub_input_batch = {k: v[init:split, :] if k not in ['inputs_embeds', 'retain_gradient'] else v for k, v in
                               input_batch.items()}
            attribution = self.deeplift_compute(explainer, sub_input_batch)
            result_attribution[init: split, :] = attribution

            init = split
        return result_attribution


    def collect_significance_scores(self, scores_path: str) -> dict:
        """
        Method collects all types of significance scores for all documents
        :param scores_path: directory to save significance scores
        :return:
        """
        progress_bar = tqdm(
            iterable=enumerate(self.dataset),
            desc=f"{self.parameters['process_name']}: Registering importance scores for ",
            total=self.dataset.__len__(),
            position=0,
            leave=True,
            file=sys.stdout
        )
        fpath = os.path.join(scores_path, f'significance_scores.pickle')
        folders = ['importance', 'lime', 'shap']
        metadata_results = dict()
        if not os.path.exists(fpath):
            for idx, datapoint in progress_bar:
                current_dict = dict()
                sample = datapoint['instance']
                if self.parameters['limit_eval']:
                    if sample['itemid'] not in self.limit_ids:
                        continue
                check_path = os.path.join(os.path.join(scores_path, f'lime_scores'), f"sample_{sample['itemid']}_lime.pickle")
                if not os.path.exists(check_path):
                    continue
                progress_bar.update(1)

                for folder in folders:
                    result_folder = os.path.join(scores_path, f'{folder}_scores')


                    file_path = os.path.join(result_folder, f"sample_{sample['itemid']}_{folder}.pickle")
                    with open(file_path, 'rb') as current_file:
                        current_data = pickle.load(current_file)

                    if folder == 'importance':
                        current_dict = {k if k != 'normalized_ig' else 'ig': v for k, v in current_data.items()}
                    else:
                        current_dict[folder if folder == 'lime' else 'deeplift'] = current_data
                metadata_results[sample['itemid']] = current_dict
            with open(fpath, 'wb') as check_file:
                pickle.dump(metadata_results, check_file)
        with open(fpath, 'rb') as check_file:
            metadata_results = pickle.load(check_file)
        return metadata_results

    def __main__(self, split: str, set_priority=False) -> None:
        """
        Method is used to register importance scores for all documents
        :param split: which split to be used for scoring [train, dev, test]
        :return: None
        """
        gt_info = 'gt' if self.parameters['access_to_gt'] else 'pred'
        self.dataset.set_split(split)
        self.significance_path = os.path.join(self.config.specific_experiment_path_isr, f'result_weights_{gt_info}')
        self.check_dir(self.significance_path)
        scores_path = os.path.join(self.significance_path, 'scores')
        result_path = os.path.join(self.significance_path, 'results')
        self.check_dir(scores_path)
        self.check_dir(result_path)
        self.extract_importance(scores_path, set_priority=set_priority)
        self.extract_lime_scores(scores_path)
        self.extract_deeplift_values(scores_path)

