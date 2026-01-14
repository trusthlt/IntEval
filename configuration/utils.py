import argparse
import os

def set_parameters()->argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=False, default='../data/dataset',
                        help='Directory for datasets')
    parser.add_argument('--downstream_path', required=False, default='../downstream',)
    parser.add_argument('--is_anonym', required=False, action='store_true',
                        help='Whether the dataset is anonymized')
    parser.add_argument('--is_singular', required=False, action='store_true',
                        help='Whether the dataset includes only single article violation cases')
    # multi-class: single case might have several violations; multi-label: each case has single violation, but there are several cases
    parser.add_argument('--task', required=False, default='existence', choices=['existence', 'multiclass', 'multilabel'],
                        help='The task to run')
    parser.add_argument('--weight_decay', type=float, default=1e-3, required=False,
                        help='The weight decay to use')
    parser.add_argument('--article_id', required=False, default='6',
                        help="Article that task will be specified for")
    parser.add_argument('--limit_length', required=False, action='store_true',
                        help='Whether the dataset is limited to a maximum length')
    parser.add_argument('--process_name', required=False, default='experiment_name',
                        help='The name of the process to track the job in the cluster')
    parser.add_argument('--seed', required=False, default=42,
                        help='Random seed to keep experiments reproducible')
    parser.add_argument('--gpu_ids', required=False, nargs="*",
                        help='Gpu IDs to be used')
    parser.add_argument('--is_mac', required=False, action='store_true',
                        help='If true, run on macOS')
    parser.add_argument('--model_path', required=False, default='legal_bert',
                        help='Model checkpoint path')
    parser.add_argument('--num_epochs', required=False, default=5, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', required=False, default=8, type=int,
                        help='Batch size')
    parser.add_argument('--learning_rate_bert', required=False, default=1e-5, type=float,
                        help='The learning rate for bert')
    parser.add_argument('--learning_rate_classifier', required=False, default=1e-4, type=float,
                        help='Learning rate for classifier')
    parser.add_argument('--context_size', required=False, type=int, default=8096,
                        help='The context size for the classifier')
    parser.add_argument('--experiment_num', required=False, default=1,
                        help='The experiment id to be run')
    parser.add_argument('--patience', required=False, default=3,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--load_f1', required=False, action='store_true',
                        help='If true, load the the best model according to f1 score, otherwise it uses accuracy')
    parser.add_argument('--resume', required=False, action='store_true',
                        help='If true, resume the training of the model from the last checkpoint')
    parser.add_argument('--drop_rate', required=False, default=0.0, type=float,
                        help='The dropout rate for the classifier')
    parser.add_argument('--train', action='store_true', required=False,
                        help='If true, run the training of the model')
    parser.add_argument('--test', action='store_true', required=False,
                        help='If true, run the testing of the model')
    parser.add_argument('--second_article_id', required=False, default='6',)
    parser.add_argument('--extract_isr', required=False, action='store_true',
                        help='Specifies whether to extract with ISR or not')
    parser.add_argument('--extract_marc', required=False, action='store_true',
                        help='Specifies whether to extract with MaRC or not')
    parser.add_argument('--access_to_gt', required=False, action='store_true',
                        help='Specifies whether ground truth values will be used for extraction or predictions')
    parser.add_argument('--model_experiments_path', required=False, default='../downstream',
                        help='Directory for results')
    parser.add_argument('--isr_experiments_path', required=False, default='../isr',
                        help='Directory for isr experiment results')
    parser.add_argument('--marc_experiments_path', required=False, default='../marc',
                        help='Directory for isr experiment results')
    parser.add_argument('--limit_eval', required=False, type=int, default=0,
                        help='Experimental reasons only, if you need to evaluate only limited amount of data')
    parser.add_argument('--min_length_rationales_isr', required=False, default=5, type=int,
                        help='Minimum length language ranges in ISR')
    return parser.parse_args()


def get_parameters() -> dict:
    parameters = dict()
    params_namespace = set_parameters()
    for argument in vars(params_namespace):
        parameters[argument] = getattr(params_namespace, argument)
    return parameters