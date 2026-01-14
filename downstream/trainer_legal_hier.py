import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../downstream')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configuration')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from tqdm import tqdm

from hier_legal_bert import HierarchicalLEGALBERT
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

import pickle
import pandas as pd

from data_modelling import DataLoadECHR
from config import Configuration

class ModelManager:
    def __init__(self, parameters, configuration_object, dataset_object):
        self.parameters = parameters
        self.configuration_object = configuration_object
        self.dataset_object = dataset_object
        self.num_labels = len(self.dataset_object.lab2id)
        self.model = None
        self.set_model()
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()
        self.criterion, self.class_weights = self.set_loss_function()

    def hierarchical_legal_bert(self):
        self.model = HierarchicalLEGALBERT(
            self.parameters, self.configuration_object, num_labels=self.num_labels
        ).to(self.configuration_object.device)
        self.model.apply(lambda x: self.set_dropout(x, drop_rate=self.parameters['drop_rate']))

    def set_dropout(self, x, drop_rate):
        if isinstance(x, nn.Dropout):
            x.p = drop_rate

    def set_model(self):
        self.hierarchical_legal_bert()

    def set_optimizer(self):
        return AdamW(
            [
                {'params': self.model.core_model.parameters(), 'lr': self.parameters['learning_rate_bert']},
                {'params': self.model.classifier.parameters(), 'lr': self.parameters['learning_rate_classifier']}
            ], weight_decay=self.parameters['weight_decay']
        )

    def set_scheduler(self):
        return CosineAnnealingLR(self.optimizer, T_max=100)

    def set_loss_function(self):
        return CrossEntropyLoss(), None



class TrainerObject:
    def __init__(self, parameters: dict, configuration_object: Configuration, model_manager: ModelManager, dataloader: DataLoadECHR):
        self.parameters = parameters
        self.configuration_object = configuration_object
        self.model = model_manager.model
        self.loss_function = model_manager.criterion
        self.optimizer = model_manager.optimizer
        self.scheduler = model_manager.scheduler
        self.dataloader = dataloader
        self.dataloader.class_weights = model_manager.class_weights
        self.experiment_path = self.configuration_object.specific_experiment_path_model

    def check_dir(self, directory: os.path) -> None:
        """
        Method to check if a directory exists and creates if it does not.
        :param directory: The directory to check.
        :return: None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def step_process(self, batch: dict, train: bool=True) -> tuple:
        """
        Method to perform one step of the training process.
        :param batch: dictionary containing the batch information (e.g., input ids, attention_mask, labels);
        :param train: Specifies whether to perform the training or not.
        :return: tuple of:
            Loss value after the step is performed;
            output of the model: Logits
        """
        batch_info = {
            "input_ids": batch['input_ids'].squeeze(1).to(self.configuration_object.device),
            "attention_mask": batch['attention_mask'].squeeze(1).to(self.configuration_object.device),
            "token_type_ids": batch['token_type_ids'].squeeze(1).to(self.configuration_object.device),
            "inputs_embeds": None,
            "retain_gradient": False,
            "extra_info": batch['extra_info']
        }
        output, _ =  self.model(**batch_info)

        targets = batch['labels'].to(self.configuration_object.device)
        loss = self.loss_function(output, targets)
        if train:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return loss.item(), output

    def compute_accuracy(self, batch_data: dict, predictions: torch.Tensor) -> tuple:
        """
        Method computes the accuracy of the predictions
        :param batch_data: batch data for process (training, test or validation)
        :param predictions: predictions of the model
        :return: tuple of:
            - label output of the model (neither distribution, nor logits);
            - ground truth label of the model;
            - accuracy of the prediction (sum of correct predictions)
        """
        if len(batch_data['itemid']) > 1:
            targets = batch_data['labels'].tolist()
        else:
            targets = batch_data['labels'].tolist() # binary / multiclass (most likely)

        outputs = torch.argmax(predictions, dim=-1).tolist()
        accuracy = sum([t == p for t, p in zip(targets, outputs)])
        return outputs, targets, accuracy

    def train(self) -> None:
        """
        Method to perform the training process.
        :return: None
        """
        desc_info = f"{self.parameters['process_name']}; {self.parameters['seed']}"
        max_f1 = 0
        epoch_no_improve = self.parameters['patience']
        epoch_idx = 0
        if self.parameters['resume']:
            epoch_num = self.load_model(resume=True)
            epoch_idx = epoch_num
            self.evaluate()

        for epoch in range(epoch_idx, self.parameters['num_epochs']):
            self.model.train()

            epoch_accuracy = 0
            epoch_loss = 0
            avg_accuracy = 0
            avg_epoch_loss = 0
            train_loader = self.dataloader['train']

            train_instances = len(train_loader.dataset)

            print(f'{20 * "<"}Train{20 * ">"}')
            progress_bar = tqdm(
                desc=f"{desc_info} => Loss: None, Accuracy: None",
                total=len(train_loader),
                position=0,
                leave=True
            )

            for batch_idx, batch in enumerate(train_loader):

                self.optimizer.zero_grad()
                step_loss, predictions = self.step_process(batch, train=True)
                epoch_loss += step_loss
                _, _, step_accuracy = self.compute_accuracy(batch, predictions)

                epoch_accuracy += step_accuracy
                avg_epoch_loss = epoch_loss / len(train_loader)
                avg_accuracy = epoch_accuracy / train_instances
                progress_bar.desc = (f"{desc_info} => "
                                     f"Epoch: {epoch + 1} / {self.parameters['num_epochs']} Loss: {avg_epoch_loss:.4f} "
                                     f"Accuracy: {avg_accuracy: .4f}")
                progress_bar.update(1)

            torch.cuda.empty_cache()

            eval_results, outcome_dict = self.evaluate(test=False)
            eval_results.update({
                'train_loss': avg_epoch_loss,
                'train_accuracy': avg_accuracy,
                'epoch': epoch + 1
            })
            f1_ = eval_results['f1_score']
            self.save_results(eval_results, outcome_dict)
            if max(f1_, max_f1) == max_f1:
                progress_bar.set_description(f'{desc_info} => No improvement, early stopping process is initiated!')
                epoch_no_improve -= 1
            else:
                max_f1 = f1_
                epoch_no_improve = self.parameters['patience']

            if not epoch_no_improve:
                progress_bar.set_description(f'{desc_info} => Early stopping occurred!')
                break
            progress_bar.close()

    def evaluate(self, test: bool = False) -> dict:
        """
        Method is used to evaluate the model. Dataset is chosen according to "test" variable
        :param test: boolean flag for testing "test" or "validation" datasets
        :return: dictionary of evaluation results
        """
        self.model.eval()
        task_name, ds_name = ('Test', 'test') if test else ('Validation', 'val')
        loader = self.dataloader[ds_name]
        num_instances = len(loader.dataset)
        desc_info = f"{self.parameters['process_name']}; {self.parameters['seed']}"

        print(f'{20 * "<"}{task_name}{20 * ">"}')
        progress_bar = tqdm(
            desc=f"Loss: None, Accuracy: None",
            total=len(loader),
            position=0,
            leave=True
        )

        with torch.no_grad():
            dev_loss = 0
            dev_accuracy = 0
            prediction_list = list()
            idx_list = list()
            target_list = list()
            avg_loss = 0
            avg_accuracy = 0

            for batch_idx, batch in enumerate(loader):

                progress_bar.update(1)
                step_loss, predictions = self.step_process(batch, train=False)
                dev_loss += step_loss
                outputs, targets, step_accuracy = self.compute_accuracy(batch, predictions)

                dev_accuracy += step_accuracy

                idx_list.extend(batch['itemid'])
                prediction_list.extend(outputs)
                target_list.extend(targets)

                avg_loss = dev_loss / len(loader)
                avg_accuracy = dev_accuracy / num_instances

                progress_bar.desc = (f"{desc_info}, {self.parameters['experiment_num']}, => "
                                     f"Loss: {avg_loss: .4f} Accuracy: {avg_accuracy: .4f}")
            outcome = {
                'itemid': idx_list,
                'predictions': prediction_list,
                'targets': target_list,
            }

            progress_bar.close()
        macro_f1 = f1_score(target_list, prediction_list, average='macro')

        eval_results = {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'f1_score': macro_f1
        }

        return eval_results, outcome

    def save_results(self, results_data: dict, outcome_dict: dict) -> None:
        """
        Saving the model performance outcome
        :param results_data: dictionary of the results (e.g., loss, accuracy, f1_score)
        :param outcome_dict: dictionary of the prediction vs targets for validation and test sets (depending on the configuration)
        :return: None
        """
        results_folder = os.path.join(self.experiment_path, 'training_results')
        self.check_dir(results_folder)
        eval_predictions_path = os.path.join(self.experiment_path, 'predictions')
        self.check_dir(eval_predictions_path)
        models_folder = os.path.join(self.experiment_path, 'checkpoints')
        self.check_dir(models_folder)

        if not os.path.exists(os.path.join(self.experiment_path, 'config.pickle')):
            pickle.dump(self.parameters, open(os.path.join(self.experiment_path, 'config.pickle'), 'wb'))

        file_name = os.path.join(results_folder, f"results_{results_data['epoch']}.pickle")

        with open(file_name, 'wb') as results_file:
            pickle.dump(results_data, results_file)

        model_name = os.path.join(models_folder, f"model_{results_data['epoch']}.pt")

        checkpoint_dict = {
            'epoch': results_data['epoch'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_metric': results_data['f1_score']
        }

        guaranteed_checkpoint_path = os.path.join(models_folder, f'model_{results_data["epoch"]}.pickle')
        pickle.dump(checkpoint_dict, open(guaranteed_checkpoint_path, 'wb'))
        torch.save(checkpoint_dict, model_name)

        fpath = os.path.join(eval_predictions_path, f"predictions_{results_data['epoch']}.csv")
        df = pd.DataFrame(outcome_dict)
        df.to_csv(fpath, index=False)

        print(f'Model with following results was saved to {model_name}:\n')
        for result, value in results_data.items():
            print(f'{result} ==> {value: .4f}')
        print('<<<<<<<<<<>>>>>>>>>>>>>>>>>')

    def get_results(self) -> dict:
        """
        Method is used to get the results of the experiment
        :return: dictionary of results
        """
        results_path = os.path.join(self.experiment_path, 'training_results')
        results_dict = dict()
        for file_path in os.listdir(results_path):

            result_file = os.path.join(results_path, file_path)
            with open(result_file, 'rb') as result_data:
                current_dict = pickle.load(result_data)
            if not results_dict:
                results_dict = {k: [v] for k, v in current_dict.items()}
                continue
            for k in results_dict.keys():
                results_dict[k].append(current_dict[k])
        return results_dict

    def load_best_metric(self, resume=False) -> tuple:
        """
        Method is used to load the best metric (predefined)
        :return: tuple that includes epoch with the best value of the chosen metric and the value itself
        """
        results = self.get_results()
        metric = 'f1_score' if self.parameters['load_f1'] else 'accuracy'

        if not resume:
            max_val = max(results[metric])
            print(f'The best metric is {metric}, with value {max_val:.4f}')
            index_value = results[metric].index(max_val)
        else:
            max_num = len(results[metric])
            index_value = results['epoch'].index(max_num)
            max_val = results[metric][index_value]
            input(results['epoch'][index_value])

        return results['epoch'][index_value], max_val

    def load_model(self, resume=False) -> int:
        """
        Method loads the best model which was chosen with given metric
        :return: None
        """
        epoch, max_val = self.load_best_metric(resume=resume)

        metric = 'f1_score' if self.parameters['load_f1'] else 'accuracy'
        print(f'Best epoch was chosen as {epoch} with {metric}: {max_val:.4f}')

        model_path = os.path.join(self.experiment_path, f'checkpoints/model_{epoch}.pt')
        checkpoints = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoints['model_state_dict'])

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device('cuda')#self.configuration_object.device
            ),
            strict=False
        )
        return epoch

    def test_model(self, save=False, direct_test=False) -> dict:
        """
        Method is used to test the model
        :return: dictionary of the results
        """
        if not direct_test:
            self.load_model()
        test_results, outcome_dict = self.evaluate(test=True)
        print(f'Test outcome:{test_results["f1_score"]}')
        if save:
            test_predictions_path = os.path.join(self.experiment_path, 'predictions')
            self.check_dir(test_predictions_path)
            fpath = os.path.join(test_predictions_path, f"predictions_test.csv")
            df = pd.DataFrame(outcome_dict)
            df.to_csv(fpath, index=False)
            print(f'Prediction results saved to {fpath}')
