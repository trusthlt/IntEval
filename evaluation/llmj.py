import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import pickle
import pandas as pd
import json
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configuration')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../downstream')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../marc')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../isr')))
from typing import Union
from config import Configuration
from extract import ExtractionISR
from rationale_creator import RationaleCreateObj



class LLMJudge:
    def __init__(self, parameters: dict, config_obj: Configuration, extractor: Union[ExtractionISR, RationaleCreateObj], extractor_name: str, expert_eval: os.path):
        self.parameters = parameters
        self.config_obj = config_obj
        self.extractor = extractor
        self.expert_eval_path = expert_eval
        self.extractor_name = extractor_name
        self.output_dir = None
        self.output_expert_judge = None

    def check_dir(self, directory: os.path) -> None:
        """
        Method is used to check if a directory exists. If not, it creates the directory.
        :param directory:
        :return: None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_rationales_folder(self) -> tuple:
        """
        Method is used to get the rationales folder. Method will create the output folder, as well.
        :return: tuple of:
            - input folder, where the rationales are stored.
            - output folder, where the LLM-outputs will be saved.
        """
        exp_path = self.extractor.significance_path
        result_folder = os.path.join(exp_path, 'results')
        if self.extractor_name == 'marc':
            rationales_folder = os.path.join(result_folder, 'rationale_outputs')
            output_folder = os.path.join(result_folder, 'judge_outputs')
            self.check_dir(output_folder)
        elif self.extractor_name == 'isr':
            length_info = self.parameters['min_length_rationales_isr']
            rationales_folder = os.path.join(result_folder, 'flexible', f'rationale_outputs_{length_info}')
            output_folder = os.path.join(result_folder, 'flexible', 'judge_outputs')
            self.check_dir(output_folder)
        else:
            raise NotImplementedError('Unknown extractor type')
        return rationales_folder, output_folder

    def collect_rationales(self) -> dict:
        """
        Method is used to collect rationales from an experiment.
        :return: dictionary of rationales with the given article id.
        """
        rationales_folder, self.output_dir = self.get_rationales_folder()
        documents = {
            6: ['001-94578', '001-120512', '001-201325', '001-92582', '001-79415'],
            8: ['001-191714', '001-219778', '001-175482', '001-228385', '001-145215']
        }
        rationales = dict()
        for article, doc_ids in documents.items():
            rationales.update({article: dict()})
            for doc_id in doc_ids:
                file_name = f'sample_{doc_id}_rationales.txt'
                file_path = os.path.join(rationales_folder, file_name)
                with open(file_path, 'r') as f:
                    rationales_list = f.readlines()
                rationales[article].update({doc_id: rationales_list})
        return rationales

    def legal_source(self) -> dict:
        """
        We use article 6 and 8 in our experiments. We provide articles to the LLMs as background information.
        :return: dictionary of legal sources.
        """
        return {
            6: "Article 6, Right to a fair trial:\n"
                 "1. In the determination of his civil rights and obligations or of any criminal charge against him, everyone is entitled to a fair and public hearing within a reasonable time by an independent and impartial tribunal established by law. Judgment shall be pronounced publicly but the press and public may be excluded from all or part of the trial in the interests of morals, public order or national security in a democratic society, where the interests of juveniles or the protection of the private life of the parties so require, or to the extent strictly necessary in the opinion of the court in special circumstances where publicity would prejudice the interests of justice. "
                 "2. Everyone charged with a criminal offence shall be presumed innocent until proved guilty according to law. \n"
                 "3. Everyone charged with a criminal offence has the following minimum rights: \n"
                 "a) to be informed promptly, in a language which he understands and in detail, of the nature and cause of the accusation against him; \n"
                 "b) to have adequate time and facilities for the preparation of his defence; \n"
                 "c) to defend himself in person or through legal assistance of his own choosing or, if he has not sufficient means to pay for legal assistance, to be given it free when the interests of justice so require; \n"
                 "d) to examine or have examined witnesses against him and to obtain the attendance and examination of witnesses on his behalf under the same conditions as witnesses against him; \n"
                 "e) to have the free assistance of an interpreter if he cannot understand or speak the language used in court. \n",
            8: "Article 8, Right to respect for private and family life: \n"
                 "1. Everyone has the right to respect for his private and family life, his home and his correspondence. \n"
                 "2. There shall be no interference by a public authority with the exercise of this right except such as is in accordance with the law and is necessary in a democratic society in the interests of national security, public safety or the economic well-being of the country, for the prevention of disorder or crime, for the protection of \n"
                 "health or morals, or for the protection of the rights and freedoms of others. \n",
        }

    def create_prompt_sufficiency(self, article_id: int, examples: Union[list, None]=None):
        """
        Method is used to create the sufficiency prompt
        :param article_id: article id for the document to be analyzed
        :param examples: None for the single shot, list for the few-shot examples
        :return: tuple of:
            - prompt: Body of the prompt, which contains instruction and legal background;
            - respond: Instruction on answer type
        """
        instruction = (
            "You are a legal expert on European Court of Human Rights decisions. Given the rationales and article(s), decide whether they are sufficient or not to decide on the existence of a violation of the given article. Also output your own confidence as High, Medium, or Low."
             "High: You are very certain of your decision; the rationales leave little room for doubt, and new information is unlikely to change it."
             "Medium: You are moderately certain; the rationales justify your decision but have gaps, so new information might change it."
             "Low: weak or fragile support; You are uncertain; the rationales weakly support your decision, and even small new/contradictory info could flip it."
        )
        article_of_choice = self.legal_source()[article_id]
        if examples:
            for idx, example in enumerate(examples):
                example_text = (f'\nExample {idx + 1}: \n**Rationales**: {example["rationales"]} \n'
                                f'**Answer**: {example["answer"]} \n**Confidence** {example["confidence"]} \n')
                instruction += example_text

        respond = (
            "Respond **EXACTLY** in this format, nothing else based on the instructions you will be provided with:"
            "**Answer:** Sufficient / Insufficient"
            "**Confidence:** High / Medium / Low"
            "**Explanation:** [Exactly 3 sentences explaining your reasoning, where the last sentence includes your answer and confidence level.]"
        )

        prompt = f'{instruction} \n Violated article: {article_of_choice} \n'
        return prompt, respond


    def create_prompt_support(self, article_id: int, examples: Union[list, None]=None) -> tuple:
        """
            Method is used to create the support prompt
            :param article_id: article id for the document to be analyzed
            :param examples: None for the single shot, list for the few-shot examples
            :return: tuple of:
                - prompt: Body of the prompt, which contains instruction and legal background;
                - respond: Instruction on answer type
        """
        instruction = (
         )
        article_of_choice = self.legal_source()[article_id]

        if examples:
            for idx, example in enumerate(examples):
                example_text = (f'\nExample {idx + 1}: \n**Rationales**: {example["rationales"]} \n'
                                f'**Answer**: {example["answer"]} \n**Confidence** {example["confidence"]} \n')
                instruction += example_text
        respond = (
            "Respond **EXACTLY** in this format, nothing else based on the instructions you will be provided with:"
            "**Answer:** Supports / Does Not Support)"
            "**Confidence:** High / Medium / Low"
            "**Explanation:** [Exactly 3 sentences explaining your reasoning, where the last sentence includes your answer and confidence level.]"
        )

        prompt = f'{instruction} \n Violated article: {article_of_choice} \n'
        return prompt, respond


    def prompt_single(self, prompt_info: dict, model_name: str) -> list:
        """
        Method is used to construct the single-shot prompt
        :param prompt_info: dictionary with information about the prompt
        :param model_name: name of the model to be used as a judge
        :return: list of messages, which differs according to the chosen model
        """
        instruction = f'{prompt_info["prompt"]} '
        query = f"Rationales are as follows: \n {prompt_info['rationales']} \n {prompt_info['respond']}"
        if model_name == 'llama':
            messages = [
                {'role': 'system', 'content': instruction },
                {'role': 'user', 'content': query},
            ]

        elif model_name == 'mistral':
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": query},
            ]

        elif model_name == 'saullm':
            messages = [
                {'role': 'user', 'content': instruction + '\n' + query},
            ]
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        return messages

    def pipeline_setup(self, model_name: str, temperature: float):
        """
        Method configures pipeline for prompting
        :param model_name: name of the model to be used as a judge
        :param temperature: temperature of the model for the experiment
        :return: pipeline object for LLM-as-a-judge setuo
        """
        if model_name == 'llama':
            model_type = 'meta-llama/Llama-3.1-8B-Instruct'

        elif model_name == 'mistral':
            model_type = 'mistralai/Mistral-7B-Instruct-v0.2'
        elif model_name == 'saullm':
            model_type = 'Equall/Saul-7b-Instruct-v1'
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_type)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_type, device_map='auto')

        return pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    def single_shot_judge(self, rationales: dict, temperature: float, is_support: bool=True, output_path: os.path=None) -> tuple:
        """
        Method performs LLM-as-a-judge in a single shot configuration
        :param rationales: dictionary of rationales
        :param temperature: temperature of the model for the experiment
        :param is_support: boolean indicating whether support or sufficiency
        :param output_path: output path to save the output
        :return: tuple of:
            - answers: dictionary of the outcomes for all models in the given configurations;
            - info_list: list of tracker to check which documents are missing (in case of malfunction)
        """
        answers = {
            'mistral': dict(),
            'saullm': dict(),
            'llama': dict(),
                   }
        info_list = list()
        for article_id, rationale_data in rationales.items():

            for model in answers.keys():
                model_answers = os.path.join(output_path, f'{model}_{article_id}.pickle' )
                if not os.path.exists(model_answers):
                    answers[model].update({article_id: dict()})
                    if is_support:
                        prompt_input, respond_input = self.create_prompt_support(article_id)
                    else:
                        prompt_input, respond_input = self.create_prompt_sufficiency(article_id)
                    print(f'<<< Model {model} {article_id} >>>')
                    default_description = f'Evaluating with {model}'
                    tqdm_iter = tqdm(rationale_data.items(), total=len(rationale_data), desc=default_description)
                    count = 0
                    pipe = self.pipeline_setup(model_name=model, temperature=temperature)
                    print(f'<<<<<<<<<<<<<<<<< {model} was initialized >>>>>>>>>>>>>>>>>>>')
                    for doc_id, rationales_ in tqdm_iter:

                        tqdm_iter.set_description(desc=f'{default_description}, doc_id: {doc_id}')
                        rationales_text = ''.join(rationales_)

                        # prompt = prompt_support + f'Rationales: {rationales} \n' + respond_support
                        prompt = {
                            'prompt': prompt_input, 'respond': respond_input,
                            'rationales': rationales_text
                        }
                        input_prompt = self.prompt_single(prompt, model_name=model)
                        tqdm_iter.set_description(desc=f'{default_description}, input prompt was generated')

                        if model == 'saullm':
                            prompt_saul = pipe.tokenizer.apply_chat_template(input_prompt, add_generation_prompt=True, tokenize=False)
                            outputs = pipe(prompt_saul, max_new_tokens=256)
                        elif model == 'llama':
                            outputs = pipe(input_prompt, max_new_tokens=256)
                        elif model == 'mistral':
                            outputs = pipe(input_prompt, max_new_tokens=256)
                        else:
                            raise ValueError(f"Invalid model name: {model}")
                        count += 1


                        output = outputs[0]['generated_text']
                        answers[model][article_id][doc_id] = output

                    pickle.dump(answers[model][article_id], open(model_answers, 'wb'))
                    print(f'<<{default_description} --> Results for article {article_id} with model {model} were saved>>')

                info_str = f'model_{model}_article_{article_id}_all_docs'
                info_list.append(info_str)
                print(f'Results for article {article_id} with model {model} were loaded>>')

                answers[model][article_id] = pickle.load(open(model_answers, 'rb'))



        return answers, info_list

    def process_single_shot(self, temp: float, rationales: dict, support: bool, is_expert_a: bool=False):
        """
        Method saves the outcomes of the given setup to the text files
        :param temp: temperature of the model for the experiment
        :param rationales: dictionary of rationales
        :param support: boolean indicating whether support or sufficiency
        :param is_expert_a: boolean indicating whether rationales belong to the expert or extraction techniques
        :return:
        """
        subfolder = os.path.join(self.output_expert_judge if is_expert_a else self.output_dir, 'single_shot', 'support' if support else 'sufficiency', f'temperature_{temp}')

        self.check_dir(subfolder)
        judge_results, info_list = self.single_shot_judge(rationales, temperature=temp, is_support=support, output_path=subfolder)

        txtfile = open(os.path.join(self.output_dir, f'{temp}_{"supp" if support else "suff"}_single.txt'), 'w')

        for each in info_list:
            txtfile.write(each)
        txtfile.close()
        self.save_text_files(judge_results, subfolder)




    def save_text_files(self, judge_results: dict, subfolder: os.path) -> None:
        """
        Method saves the outcomes of the given setup to the text files
        :param judge_results: dictionary that contains the results of the experiment
        :param subfolder: path to the subfolder where the results will be saved
        :return: None
        """
        for model_name, model_results in judge_results.items():
            model_sub_path = os.path.join(subfolder, model_name)
            self.check_dir(model_sub_path)
            for article_id, eval_results in model_results.items():
                article_sub_path = os.path.join(model_sub_path, f'article_{article_id}')
                self.check_dir(article_sub_path)
                for doc_id, output in eval_results.items():
                    file_path = os.path.join(article_sub_path, f'LLM_eval_{doc_id}.txt')
                    if not os.path.exists(file_path):
                        out = self.process_output(output, model_name)
                        text_file = open(file_path, 'w')
                        text_file.write(out)
                    # if you want to see the outputs, comment out the following lines
                    # text_file = open(file_path, 'r')
                    # out = text_file.readlines()
                    # print(doc_id, model_name, article_id)
                    # print(out)

                    # print('<<<<<==============>>>>>>')
        print(f'All outputs were saved to {subfolder}')

    def expert_eval(self, is_expert_a: bool=False) -> dict:
        """
        We collect expert B's evaluation results with this method to compare LLMs with respect to these results
        :param is_expert_a: boolean indicating whether rationales belong to the expert or extraction techniques
        :return: dictionary that contains the evaluation results of the expert B
        """
        if self.extractor_name == 'marc':
            file_name = 'expert_B_analysis_t1.csv'
        elif self.extractor_name == 'isr':
            file_name = 'expert_B_analysis_t2.csv'
        elif is_expert_a:
            file_name = 'expert_B_analysis_t3.csv'
        else:
            raise ValueError('There is not such configuration')

        file_path = os.path.join(self.expert_eval_path, 'annotations', 'expert_b_eval', file_name)

        data = pd.read_csv(file_path)

        documents = {
            6: ['001-94578', '001-120512', '001-201325', '001-92582', '001-79415'],
            8: ['001-191714', '001-219778', '001-175482', '001-228385', '001-145215']
        }
        result_dict = {6: dict(), 8: dict()}
        confidence_map = {
            'H': 'High', 'M': 'Medium', 'L': 'Low'
        }
        for article_id, document_ids in documents.items():
            for document_id in document_ids:
                subframe = data[data['itemid'] == document_id]
                sufficiency, suff_confidence = subframe['sufficiency'].item().split('-')
                support, supp_confidence = subframe['support'].item().split('-')

                result_dict[article_id].update({
                    document_id: {
                        'sufficiency': sufficiency,
                        'sufficiency_confidence': confidence_map[suff_confidence],
                        'support': support.replace('Global', ''),
                        'support_confidence': confidence_map[supp_confidence]
                    }
                })
        return result_dict

    def process_few_shot(self, temp: float, rationales: dict, support: bool, is_expert_a: bool=False):
        """
        Method saves the outcomes of the given setup to the text files --- for the few shot experiments
        :param temp: temperature of the model for the experiment
        :param rationales: dictionary of rationales
        :param support: boolean indicating whether support or sufficiency
        :param is_expert_a: boolean indicating whether rationales belong to the expert or extraction techniques
        :return: None
        """

        subfolder = os.path.join(self.output_expert_judge if is_expert_a else self.output_dir, 'few_shot', 'support' if support else 'sufficiency',
                                 f'temperature_{temp}')
        expert_evaluations = self.expert_eval(is_expert_a=is_expert_a)
        self.check_dir(subfolder)
        judge_results, info_list = self.few_shot_judge(rationales, support, expert_evaluations, temp, subfolder)
        print(temp, 'supp' if support else 'suff')

        txtfile = open(os.path.join(self.output_dir, f'{temp}_{"supp" if support else "suff"}_few.txt'), 'w')

        for each in info_list:
            txtfile.write(each)
        txtfile.close()
        self.save_text_files(judge_results, subfolder)


    def collect_few_shot_examples(self, doc_id: str, support: bool, rationales_per_article: dict, eval_results_per_article: dict) -> list:
        """
        Method collects rationale examples for the few shot experiments
        :param doc_id: document id that we collect rationales for
        :param support: boolean indicating whether support or sufficiency
        :param rationales_per_article: dictionary that contains all the rationales per article
        :param eval_results_per_article: expert evaluation results
        :return: list of dictionaries
        """
        few_shot_examples = list()
        for doc_id_example, eval_results_examples in eval_results_per_article.items():
            if doc_id == doc_id_example:
                continue

            example_rationale_list = rationales_per_article[doc_id_example]
            example = {
                "rationales": ''.join(example_rationale_list),
                "answer": eval_results_examples['support'] if support else eval_results_examples['sufficiency'],
                "confidence": eval_results_examples["support_confidence"] if support else eval_results_examples[
                    'sufficiency_confidence']
            }
            few_shot_examples.append(example)

        return few_shot_examples

    def few_shot_judge(self, rationales: dict, support: bool, expert_evaluations: dict, temperature: float, output_path: os.path):
        """
        Method performs few shot experiments
        :param rationales: dictionary that contains all the rationales per article
        :param support: boolean indicating whether support or sufficiency
        :param expert_evaluations: expert evaluation results
        :param temperature: temperature of the model for the experiment
        :param output_path: path to the output directory
        :return: tuple of:
            - answers: dictionary of the outcomes for all models in the given configurations;
            - info_list: list of tracker to check which documents are missing (in case of malfunction)
        """
        answers = {'saullm': dict(), 'llama': dict(), 'mistral': dict()}
        info_list = list()

        for article_id, eval_results_per_article in expert_evaluations.items():
            rationales_per_article = rationales[article_id]

            for model in answers.keys():
                model_answers = os.path.join(output_path, f'{model}_{article_id}.pickle' )
                print(f'<<< Model {model} {article_id} >>>')


                if not os.path.exists(model_answers):
                    answers[model].update({article_id: dict()})
                    default_description = f'Evaluating with {model}'
                    tqdm_iter = tqdm(eval_results_per_article.items(), total=len(eval_results_per_article), desc=default_description)
                    pipe = self.pipeline_setup(model_name=model, temperature=temperature)
                    print(f'<<<<<<<<<<<<<<<<< {model} was initialized >>>>>>>>>>>>>>>>>>>')
                    for doc_id, eval_results in tqdm_iter:

                        rationale_list = rationales_per_article[doc_id]

                        few_shot_examples = self.collect_few_shot_examples(doc_id, support, rationales_per_article, eval_results_per_article)
                        if support:
                            prompt_input, respond_input = self.create_prompt_support(article_id, few_shot_examples)
                        else:
                            prompt_input, respond_input = self.create_prompt_sufficiency(article_id, few_shot_examples)

                        tqdm_iter.set_description(desc=f'{default_description}, doc_id: {doc_id}')
                        rationales_text = ''.join(rationale_list)
                        # prompt = prompt_support + f'Rationales: {rationales} \n' + respond_support
                        prompt = {
                            'prompt': prompt_input, 'respond': respond_input,
                            'rationales': rationales_text
                        }
                        input_prompt = self.prompt_single(prompt, model_name=model)
                        tqdm_iter.set_description(desc=f'{default_description}, input prompt was generated')


                        if model == 'saullm':
                            prompt_saul = pipe.tokenizer.apply_chat_template(input_prompt, add_generation_prompt=True, tokenize=False)
                            outputs = pipe(prompt_saul, max_new_tokens=256)
                        elif model == 'llama':
                            outputs = pipe(input_prompt, max_new_tokens=256)
                        elif model == 'mistral':
                            outputs = pipe(input_prompt, max_new_tokens=256)
                        else:
                            raise ValueError(f"Invalid model name: {model}")

                        output = outputs[0]['generated_text']
                        answers[model][article_id][doc_id] = output
                    pickle.dump(answers[model][article_id], open(model_answers, 'wb'))
                    print(f'<<{default_description} --> Results for article {article_id} with model {model} were saved>>')
                info_str = f'model_{model}_article_{article_id}_all_docs'

                print(f'Results for article {article_id} with model {model} were loaded')
                info_list.append(info_str)

                answers[model][article_id] = pickle.load(open(model_answers, 'rb'))

        return answers, info_list

    def collect_expert_a_rationales(self) -> dict:
        """
        Collecting rationales, which were extracted by expert A
        :return: dictionary that contains all the rationales per article by expert A
        """
        expert_a_rationales_path = os.path.join(self.expert_eval_path, 'rationales')
        documents = {
            6: ['001-94578', '001-120512', '001-201325', '001-92582', '001-79415'],
            8: ['001-191714', '001-219778', '001-175482', '001-228385', '001-145215']
        }
        rationales = dict()
        for article, doc_ids in documents.items():
            rationales.update({article: dict()})
            for doc_id in doc_ids:
                file_name = f'Exp_B_{article}_{doc_id}_rationales_t3.txt'
                file_path = os.path.join(expert_a_rationales_path, file_name)
                with open(file_path, 'r') as f:
                    rationales_list = f.readlines()
                rationales[article].update({doc_id: rationales_list})

        self.output_expert_judge = os.path.join(self.expert_eval_path, 'judge_outputs')
        self.check_dir(self.output_expert_judge)

        return rationales

    def kappa_ci(self, expert: list, llm: list, n_bootstrap: int=10000, ci: float=0.95):
        """
        We compute confidence interval on the kappa score
        :param expert: expert evaluation results
        :param llm: llm evaluation results
        :param n_bootstrap: number of bootstrap samples
        :param ci: confidence interval
        :return: tuple of:
            - point_kappa: Cohen's kappa;
            - lower: lower confidence interval;
            - upper: upper confidence interval;
        """
        kappas = []
        n = len(expert)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            boot_expert = np.array(expert)[idx]
            boot_llm = np.array(llm)[idx]
            kappa = cohen_kappa_score(boot_expert, boot_llm)
            kappas.append(kappa)
        lower = np.percentile(kappas, (1 - ci) / 2 * 100)
        upper = np.percentile(kappas, (1 + ci) / 2 * 100)
        point_kappa = cohen_kappa_score(expert, llm)
        return round(point_kappa, 2), (round(lower, 2), round(upper,2))


    def compute_agreement(self, metric: str, type_shot: str) -> tuple:
        """
        Method to compute the agreement between expert evaluation results and the LLM evaluations
        :param metric: support or sufficiency metric
        :param type_shot: single or few shot evaluations
        :return: tuple of:
            - results: dictionary of agreement between expert evaluation results and LLMs
            - results_combination: inter-LLM agreement (binary setup)
        """

        data = pd.read_csv(f'../evaluation/{metric}_{type_shot}_shot.csv')
        results = {'model': list(), 'values': list()}
        results_combination = {'model': list(), 'values': list()}
        for each in ['Saul', 'LLAMA', 'Mistral']:

            kappa, limits = self.kappa_ci(data['Expert B'], data[each])
            results['model'].append(each)
            results['values'].append((kappa, limits))


        for combination in [('Saul', 'LLAMA'), ('Mistral', 'Saul'), ('Mistral', 'LLAMA')]:
            results_combination['model'].append(f'{combination[0]}-{combination[1]}')
            kappa, limits = self.kappa_ci(data[combination[0]], data[combination[1]])
            results_combination['values'].append((kappa, limits))

        return results, results_combination

    def __main__(self):
        """
        Method combines all processes in one function
        :return: None
        """
        rationales_expert_a = self.collect_expert_a_rationales()
        rationales = self.collect_rationales()
        # collects llm-as-a-judge results
        for temperature in [0.05, 0.5, 1.0]:
            for support in [True, False]:
                # techniques' rationales
                print(f'<<<<<<<<<<<<<<< {"Support" if support else "Sufficiency"}, single, {temperature} started >>>>>>>>>>>>>>>>>>')
                self.process_single_shot(temperature, rationales, support)
                print(f'<<<<<<<<<<<<<<< {"Support" if support else "Sufficiency"}, single, {temperature} finished >>>>>>>>>>>>>>>>>>')
                print(f'<<<<<<<<<<<<<<< {"Support" if support else "Sufficiency"}, few, {temperature} started >>>>>>>>>>>>>>>>>>')
                self.process_few_shot(temperature, rationales, support)
                print(f'<<<<<<<<<<<<<<< {"Support" if support else "Sufficiency"}, few, {temperature} finished >>>>>>>>>>>>>>>>>>')
                # expert a's rationales
                print(f'<<<<<<<<<<<<<<< {"Support" if support else "Sufficiency"}, single, {temperature} started -> exp A >>>>>>>>>>>>>>>>>>')
                self.process_single_shot(temperature, rationales_expert_a, support, is_expert_a=True)
                print(f'<<<<<<<<<<<<<<< {"Support" if support else "Sufficiency"}, single, {temperature} finished -> exp A >>>>>>>>>>>>>>>>>>')
                print(f'<<<<<<<<<<<<<<< {"Support" if support else "Sufficiency"}, few, {temperature} started -> exp A >>>>>>>>>>>>>>>>>>')
                self.process_few_shot(temperature, rationales, support, is_expert_a=True)
                print(f'<<<<<<<<<<<<<<< {"Support" if support else "Sufficiency"}, few, {temperature} finished -> exp A >>>>>>>>>>>>>>>>>>')

        # computes cohen's kappa for agreement values, including bootstrapping confidence interval
        results = dict()
        comb_results = dict()
        for metric in ['sufficiency', 'support']:
            types = dict()
            types_comb = dict()
            for type_shot in ['single', 'few']:
                print(f'<<-- {metric} -- {type_shot} -->>')
                types[type_shot], types_comb[type_shot] = self.compute_agreement(metric, type_shot)

            results[metric] = types
            comb_results[metric] = types_comb
            print(f'<<<<<<<<<-----{metric}----->>>>>>>>>>>>>>>')
        print(comb_results['sufficiency'])
        print(comb_results['support'])

        print(results['sufficiency'])
        for typ, val in results['sufficiency'].items():
            print(typ)
            print(pd.DataFrame(val))

        for typ, val in results['support'].items():
            print(typ)
            print(pd.DataFrame(val))
        input('<<>>')


    def process_output(self, output: Union[str, list], model_name: str) -> str:
        """
        Method to process the output of the LLMs
        :param output: either string (for saullm) or list that contains the judgment
        :param model_name: name of the model
        :return: answer of the LLM
        """
        if model_name == 'saullm':
            response = output.split("[/INST]")[-1]
        elif model_name == 'llama':
            response = output[-1]['content']
        elif model_name == 'mistral':
            response = output[-1]['content']
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        return response




