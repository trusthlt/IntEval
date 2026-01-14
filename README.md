# IntEval: Evaluation of interpretability techniques in legal domain

## Packages
- ### configuration: folder includes 2 .py files and 2 configuration files:
  - setup.py: The configuration object is initiated when you run the project and configures project's setting, automatically;
  - utilities.py: All arguments for the model setup are collected with argument parser in this file and distributed across all files;
  - environment.yml: environment configuration;
  - cuda_requirements.txt: It will install same cuda configuration as we have for this project;
---
- ### data: folder must includes raw data under echr folder:
  - you must place the provided datasets into [data/dataset](data/dataset) folder
  - data_processing folder includes following packages (the order of the items are organized as their running hierarchy):
  - process_data.py: Collects raw data files for smooth processing;
  - dataset.py: Converts processed data into torch Dataset object;
---
- ### downstream: includes 3 files, which runs the downstream binary classification task:
  - hier_legal_bert.py: includes model configuration and model wrapper objects;
  - trainer_legal_hier.py: Configuration of the training setting and saving results.
  - data_modelling.py: Processes the dataset for training as utilized in MaRC;
  - When model is run, predefined experiment folder is created. If you run again with the same experiment number, trainer object will skip training session.
---
- ### isr: Flexible **I**nstance **S**pecific **R**ationale Extraction technique's folder. It includes following packages:
  - register.py: Computes and collects importance scoring techniques' scores for the given input data;
  - extract.py: Given registered information, it extracts rationales for the chosen configuration;
  - predictors.py: It includes lime predictor and Shapley wrapper (for DeepLift);
  - When model is run, experiments folder is created, so that all results from extraction are collected here;
---
- ### marc: Includes one file:
  - rationale_creator.py: all optimization processes to extract rationales are organized by this file;
  - When model is run, experiments folder is created, so that all results from extraction are collected here;
---
- ### evaluations: includes all files to develop evaluation framework:
  - evaluation.py: Runs quantitative evaluation sessions (NormComp, NormSuff, F1-Suff and F1-Comp);
  - generate_text.py: This file extracts textual output according to the rationale extraction configuration;
  - llm_judge.py: Runs LLM-as-a-judge evaluation settings;
---
- ### expert analysis: includes expert evaluation documents:
  - annotations: expert_b_eval folder in this given folder, includes expert b's evaluation results on rationales;
  - rationales: includes rationales extracted by expert A.
---
## How to reproduce the project?
  - First of all, we highly recommend you to create new environment with the provided environment file in the configuration folder;
    ```bash
      conda env create -f configuration/environment.yml
      conda activate inteval
    ```
  - You can also install cuda configuration that we use, by running the following command
    ```bash
      pip install -r configuration/cuda_requirements.txt
    ```
  - Before going further we recommend you to change the directory to srs, before running main.py. Because of the folder structure, some issues may happen, unless you change it.
    - Once you already changed it we can go on to run the code:
  - running for training and testing first (recommended split the process into two as we do here):
    ```bash
      python main.py --process_name reproduce \
              --model_path legal_bert\
              --gpu_ids 3 \
              --experiment_num 19 \
              --article_id all \
              --context_size 512 \
              --batch_size 4 \
              --seed 42 \
              --num_epochs 1 \
              --load_f1 \
              --weight_decay 0.001 \
              --drop_rate 0.1 \
              --train \
              --test \
              --limit_eval 2
    ```
  - running for rationale extraction, quantitative evaluation and llm-as-a-judge configuration:
  ```bash
      python main.py --process_name reproduce \
              --model_path legal_bert\
              --gpu_ids 3 \
              --experiment_num 19 \
              --article_id all \
              --context_size 512 \
              --batch_size 4 \
              --seed 42 \
              --num_epochs 1 \
              --load_f1 \
              --weight_decay 0.001 \
              --drop_rate 0.1 \
              --train \
              --test \
              --limit_eval 2
```
  - reason for splitting: There might be conflict because of the dataset choice.
  - extract_marc and extract_isr are boolean variables that launch rationale extraction processes for specific technique;
  - We suggest to provide number different than 1 to the experiment_num, since experiment 1 is our results for you to check. You will not be able to run it, since it skips already-done processes;
  - gpu_ids represent cuda device id that you will run your project on (if you don't specify it is 0)
  - limit_eval stands for limited evaluation and extraction, since whole process may take few hours. For full evaluation scenario, either remove it from the command or set it to 0. Upper limit for this variable is 2998, since evaluation is done based on the test dataset.
  - Notice that, we set_priority variable in both extraction techniques to true, so it will extract rationales for the cases we used for evaluation, first. This will discard limit_eval parameter's choice.