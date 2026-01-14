import os
import pickle
import random
import numpy as np

import torch.backends.mps


class Configuration:
    def __init__(self, parameters, load=False):
        self.parameters = parameters
        self.set_visible_devices()
        self.model_results = os.path.join(self.parameters['model_experiments_path'], 'experiments')

        self.marc_results = os.path.join(self.parameters['marc_experiments_path'], 'experiments')
        self.isr_results = os.path.join(self.parameters['isr_experiments_path'], 'experiments')
        self.device = self.set_device()
        self.model_names = self.model_abbreviations()
        self.specific_experiment_path_model = None
        self.specific_experiment_path_marc = None
        self.specific_experiment_path_isr = None
        self.config_setup()

    def check_dir(self, directory: os.path) -> None:
        """
        Method is used to check if a directory exists. It will create the directory if it doesn't exist.'
        :param directory: path to directory to be checked
        :return: None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def set_visible_devices(self) -> None:
        """
        Method assigns gpus to be used for the project
        :return: None
        """
        devices_info = ','.join(self.parameters['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = devices_info
        os.environ['MASTER_PORT'] = '29503'


    def result_setup(self) -> None:
        """
        Method is used to set up the result folder for each experiment: Downstream task, MaRC and ISR.
        :return: None
        """
        self.check_dir(self.model_results)
        self.check_dir(self.marc_results)
        self.check_dir(self.isr_results)

    def config_setup(self) -> None:
        """
        Method is used to setup the configuration for each experiment.
        :return:
        """
        self.result_setup()
        self.set_folder_up()
        configuration_file = os.path.join(self.specific_experiment_path_model, 'config.pickle')
        if not os.path.exists(configuration_file):
            with open(configuration_file, 'wb') as config_data_file:
                pickle.dump(self.parameters, config_data_file)

    def set_folder_specific(self, model_name, exp_type='model') -> os.path:
        """
        Method creates the specific folder for each experiment and configures its directory structure
        :param model_name: which model was used as downstream classifier
        :param exp_type: experiment type - model, marc or isr
        :return: experiment folder for the chosen experiment number, seed value
        """

        if exp_type == 'model':
            specific_path = self.model_results
        elif exp_type == 'marc':
            specific_path = self.marc_results
        elif exp_type == 'isr':
            specific_path = self.isr_results
        else:
            raise NotImplementedError
        experiments_path = os.path.join(specific_path, f'{model_name}')

        self.check_dir(experiments_path)
        experiment_folder = os.path.join(experiments_path, f'experiment_{self.parameters["experiment_num"]}', f'seed_{self.parameters["seed"]}')
        self.check_dir(experiment_folder)
        return experiment_folder


    def set_folder_up(self) -> None:
        """
        Method creates all the folders for the experiments at once
        :return: None
        """
        chosen_model_name =self.parameters["model_path"]
        experiments_marc_path = os.path.join(self.model_results, chosen_model_name)
        self.check_dir(experiments_marc_path)
        self.specific_experiment_path_model = self.set_folder_specific(chosen_model_name, 'model')
        self.specific_experiment_path_marc = self.set_folder_specific(chosen_model_name, 'marc')
        self.specific_experiment_path_isr = self.set_folder_specific(chosen_model_name, 'isr')

    def model_abbreviations(self) -> dict:
        """
        Method returns the dictionary with all model abbreviations (Note: We kept only legal-bert as it was the best)
        :return: dictionary with all model abbreviations
        """
        models = {
            'legal_bert': 'nlpaueb/legal-bert-base-uncased',
        }
        return models

    def set_device(self) -> str:
        """
        Method sets the device to be used for the experiments and ensures the deterministic experimental setup for
        reproducibility
        :return: string to specify device name
        """
        device = 'cpu'

        if torch.cuda.is_available():
            device = 'cuda'
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

            self.set_visible_devices()
            random.seed(self.parameters['seed'])
            np.random.seed(int(self.parameters['seed']))
            torch.cuda.manual_seed(self.parameters['seed'])
            torch.cuda.manual_seed_all(self.parameters['seed'])
            os.environ['PYTHONHASHSEED'] = str(self.parameters['seed'])

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        return device



