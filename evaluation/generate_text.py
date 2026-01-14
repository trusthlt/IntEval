import os
import sys
import pickle

import pandas as pd
from datasets.utils.extract import Extractor
from sty import bg
from tqdm import tqdm
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configuration')))
# from setup import Configuration

class ExtractTextValues:
    def __init__(self, parameters_dict: dict, config_obj, tech, extractor_obj):
        self.parameters_dict = parameters_dict
        self.config_obj = config_obj
        self.tech = tech
        self.extractor = extractor_obj
        self.experiment_folder = self.set_experiment_folder()
        self.significance_path = None

    def check_dir(self, directory: os.path) -> None:
        """
        Method checks the existence of a directory. If it doesn't exist, it creates it.
        :param directory: path to directory to check
        :return: None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

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

    def textual_out(self, data: dict, out_path: os.path, rat_out_path: os.path, is_marc: bool=True):
        """
        Method generates textual output and saves it to the specific path
        :param data: output dictionary as a result of scoring
        :param out_path: output path where text with rationales will be saved
        :param rat_out_path: output path where only rationales will be saved
        :param is_marc: boolean variable specifies whether it is marc or isr
        :return: None
        """
        words = list(zip(data['weight'] if is_marc else data['mask'], [w['word'] for w in data['words']]))

        highlighted_text = list()
        non_rationale_text = list()
        for idx in range(len(words)):
            if words[idx][0] >= 0.2:
                highlighted_text.append(idx)
            else:
                non_rationale_text.append(idx)

        non_rat = [{'idx': idx, 'combination': -1} for idx in non_rationale_text]
        rationales = self.process_texts(highlighted_text)

        text_indexes = sorted(non_rat + rationales, key=lambda x: x['idx'])

        rationales_only = self.generate_rationales_only(rationales, words)
        text_output = self.generate_case_doc(text_indexes, words)

        text_file = open(out_path, 'w')
        text_file.write(text_output)
        text_file.close()
        rationale_file = open(rat_out_path, 'w')
        rationale_file.write(rationales_only)
        rationale_file.close()

    @staticmethod
    def generate_case_doc(list_of_indexes: list, words: list) -> str:
        """
        Generates text with rationales from a list of indexes and words in a suitable format for INCePTION
        :param list_of_indexes: List of index combinations for the rationales (which passes the threshold)
        :param words: list of words to generate text with
        :return: text output
        """
        prev_combination = -1
        supposed_text = list()

        for sig_word_idx, word_idx_info in enumerate(list_of_indexes):
            current_combination = word_idx_info['combination']
            current_word = words[word_idx_info['idx']][1]

            if current_combination != prev_combination:
                if prev_combination == -1: # when the previous token was non-rationale but now is rationale
                    word = f'\n|*<< {current_word}'# then new rationale starts

                else: # when the previous token was rationale but current is non-rationale
                    word = f'>>*|\n{current_word}'
            else:
                word = current_word
                if sig_word_idx == len(list_of_indexes) - 1 and current_combination != -1:
                    word += '>>*|'
            supposed_text.append(word)
            prev_combination = current_combination
        text = ' '.join(supposed_text)
        return text

    @staticmethod
    def generate_rationales_only(list_of_indexes: list, words: list) -> str:
        """
        Method generates text with only rationales from a list of indexes and words in a suitable format
        :param list_of_indexes: List of index combinations for the rationales (which passes the threshold)
        :param words: list of words to generate text with
        :return: text output (only rationales)
        """
        prev_combination = -1
        supposed_text = list()

        for sig_word_idx, word_idx_info in enumerate(list_of_indexes):
            current_combination = word_idx_info['combination']
            current_word = words[word_idx_info['idx']][1]

            if current_combination - prev_combination:
                word = f'|*<< {current_word}'
                supposed_text.append(word)
                if prev_combination != -1:

                    supposed_text[sig_word_idx - 1] = f"{supposed_text[sig_word_idx - 1]}>>*| \n"
            else:
                word = f'{current_word}'
                supposed_text.append(word)
            prev_combination = current_combination
        text = ' '.join(supposed_text)
        return text

    @staticmethod
    def process_texts(list_text: list) -> list:
        """
        Method collects indexes of the rationales and creates a dictionary for the combinations, which will be used to
        generate the text which is in useful format for INCEPTION
        :param list_text: list of indexes of the rationales (which passes the threshold)
        :return: list of dictionaries for each index
        """
        comprehensive_highlights = list()
        combination_idx = 0
        for idx in range(len(list_text)):
            if not idx:
                comprehensive_highlights.append({'idx': list_text[idx], 'combination': combination_idx})
                continue

            if list_text[idx] - list_text[idx - 1] != 1:
                combination_idx += 1


            comprehensive_highlights.append({'idx': list_text[idx], 'combination': combination_idx})

        return comprehensive_highlights

    def generate_text_forall(self) -> None:
        """
        Method is the evaluation manager for the whole generalized process, that can adapt evaluation process to the
        given technique
        :return: None
        """
        feature_name_list = [
            "flexible", "lime", "attention", "gradients", "ig", "scaled_attention", "deeplift"
        ] if self.tech == 'isr' else [""]

        for feature_name in feature_name_list:
            score_folder = os.path.join(self.experiment_folder, feature_name)
            if not feature_name:
                feature_name = 'marc'
            if not os.path.exists(score_folder):
                raise NotImplementedError(f"There is not masks for feature name {feature_name}")
            self.__main__(score_folder, feature_name)

    def __main__(self, results_folder: os.path, feature_name: str) -> None:
        """
        Method is the main operating function of the class - it performs the text extraction
        :param results_folder: Path to the folder where the results are stored
        :param feature_name: feature name stands for the technique that rationales extracted with
        :return: None
        """
        if self.tech == 'isr':
            rationale_mask_folder = os.path.join(results_folder, f'rationale_masks_min_{self.parameters_dict["min_length_rationales_isr"]}')
            extra_info = f'_{self.parameters_dict["min_length_rationales_isr"]}'
            is_marc = False
        elif self.tech == 'marc':
            rationale_mask_folder = os.path.join(results_folder, 'meta_data_marc')
            extra_info = ''
            is_marc = True
            feature_name = 'marc'
        else:
            raise NotImplementedError

        datafiles_list = os.listdir(rationale_mask_folder)
        text_out_folder = os.path.join(results_folder, f'text_outputs{extra_info}')
        self.check_dir(text_out_folder)
        rationales_out_folder = os.path.join(results_folder, f'rationale_outputs{extra_info}')
        self.check_dir(rationales_out_folder)
        self.generate_text_data(datafiles_list, text_out_folder, rationales_out_folder, rationale_mask_folder, feature_name, is_marc)

    def generate_text_data(self, list_of_data: list, output_folder: os.path, rationales_out_folder: os.path,
                           mask_folder: os.path, technique_name: str, is_marc: bool) -> None:
        """
        Method generates the text output for a given technique using rationale masks
        :param list_of_data: list of rationale mask files
        :param output_folder: text output folder
        :param rationales_out_folder: rationale output folder
        :param mask_folder: rationale mask folder
        :param technique_name: name of the technique to generate text for
        :param is_marc: is this technique marc or not
        :return: None
        """
        iterator = tqdm(iterable=list_of_data, total=len(list_of_data), desc=f'Extracting textual outputs')
        for datapoint_file in iterator:
            file_dir = os.path.join(mask_folder, datapoint_file)
            if 'tracker.pickle' in file_dir:
                continue
            with open(file_dir, 'rb') as document:
                data = pickle.load(document)

            output_file_path = datapoint_file.replace('.pickle', '.txt')
            rationales_file_path = output_file_path.replace('.txt', '_rationales.txt')
            out_dir = os.path.join(output_folder, output_file_path)
            rat_out_dir = os.path.join(rationales_out_folder, rationales_file_path)
            if not os.path.exists(rat_out_dir):
                iterator.set_description(f'{technique_name} => {output_file_path} and {rationales_file_path} are created!')
                self.textual_out(data, out_dir, rat_out_dir, is_marc)
            else:
                iterator.set_description(f'{technique_name} => {output_file_path} and {rationales_file_path} were already created!')