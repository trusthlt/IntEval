import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configuration')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../downstream')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../marc')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../isr')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation')))



import copy
from config import Configuration
from utils import *
from processing.dataset import VioSet
from processing.process_data import *
from data_modelling import DataLoadECHR
from trainer_legal_hier import TrainerObject, ModelManager
from rationale_creator import *
from extract import *
from quant_evaluation import *
from generate_text import *
from llmj import LLMJudge
import setproctitle
import pandas as pd


def __main__():
    parameters = get_parameters()
    setproctitle.setproctitle(parameters['process_name'])
    config_obj = Configuration(parameters)

    ds_obj = VioSet(parameters, config_obj, split='train')
    dataloader = DataLoadECHR(config_obj, ds_obj, parameters)

    model_manager = ModelManager(parameters, config_obj, ds_obj)
    train_obj = TrainerObject(parameters, config_obj, model_manager, dataloader)
    expert_eval_path = '../expert_analysis'

    if parameters['train']:
        train_obj.train()

    if parameters['test']:
        train_obj.test_model(direct_test=False, save=True)
        original_experiment_path = copy.deepcopy(config_obj.specific_experiment_path_model)
        for specific_article in ['6', '8']:

            current_experiment_path = os.path.join(
                original_experiment_path, f'experiments_article_{specific_article}'
            )
            parameters['article_id'] = specific_article
            config_obj.specific_experiment_path_model = current_experiment_path
            current_set = VioSet(parameters, config_obj, split='train')
            current_dataloader = DataLoadECHR(config_obj, current_set, parameters)

            model_manager = ModelManager(parameters, config_obj, ds_obj)
            model_manager.model = train_obj.model
            train_obj_specific = TrainerObject(parameters, config_obj, model_manager, current_dataloader)
            train_obj_specific.test_model(direct_test=True, save=True)

    if parameters['extract_marc']:
        rat_cr = RationaleCreateObj(
            parameters_dict=parameters,
            config_obj=config_obj,
            dataset_obj=ds_obj,
            dataloader_obj=dataloader,
            trainer_obj=train_obj
        )
        rat_cr.process_main(set_priority=True)
        q_eval = QuantitativeEvaluation(extractor_obj=rat_cr, tech_name='marc', config_obj=config_obj,
                                        parameters=parameters, dataset_obj=ds_obj)
        q_eval.evaluate_technique()
        llmj = LLMJudge(parameters, config_obj, rat_cr, 'marc', expert_eval=expert_eval_path)
        llmj.__main__()
        text_gen = ExtractTextValues(parameters, config_obj, 'marc', rat_cr)
        text_gen.generate_text_forall()

    if parameters['extract_isr']:
        extractor = ExtractionISR(
            parameters_dict=parameters,
            dataset_obj=ds_obj,
            dataloader_obj=dataloader,
            config_obj=config_obj,
            trainer_obj=train_obj
        )
        extractor.__main__(split='test', set_priority=True)
        q_eval = QuantitativeEvaluation(extractor_obj=extractor, tech_name='isr', config_obj=config_obj,
                                        parameters=parameters, dataset_obj=ds_obj)
        q_eval.evaluate_technique()

        llmj = LLMJudge(parameters, config_obj, extractor, 'isr', expert_eval=expert_eval_path)
        llmj.__main__()

        text_gen_isr = ExtractTextValues(parameters, config_obj, 'isr', extractor)
        text_gen_isr.generate_text_forall()


if __name__ == '__main__':
    __main__()

