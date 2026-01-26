# DisGeneFormer
An Attention-based Graph Transformer for Disease Gene Prioritization. 

Please refer to the original paper for additional information:

Any publication that discloses findings arising from using this source code, the model parameters, or outputs produced by those should also cite this paper.

If you encounter any problems, please open an Issue on this repository.

## Table of contents
* [Setup](#setup)
* [Usage](#usage)
    * [Train the generic model](#train-the-generic-model)
    * [Predict using the generic model](#predict-using-the-generic-model)
    * [Train the specific model](#train-the-specific-model)
    * [Predict using the specific model](#predict-using-the-specific-model)
* [Additional material](#additional-material)

## Setup

DisGeneFormer requires Python 3.11.9
The environment is setup using conda and pip packages.
If there are issues installing the pip packages when setting up the conda environment from `environment/environment.yml`, activate the conda environment and then install them separately using pip while inside the environment.
```bash
conda env create -f environment/environment.yml
conda activate DisGeneFormer_env
pip install -r environment/requirements.txt
```

## Download all models and datasets
To download all raw datasets needed to reproduce all experiments in the manuscript, use:
```bash
bash scripts/download_all_datasets.sh
```

## Reproduce all experiments
To reproduce all experiments from scratch in the manuscript, you may run the following script. Note that this includes training each model from scratch and then running inference and evaluation which may take some time. To do this faster, skip to the next step instead to download the trained model weights and only run inference and evaluation.
```bash
bash scripts/reproduce_all_experiments.sh
```

## Download all models and results
To avoid having to retrain all models from scratch to reproduce all experiments, we additionally provide all trained models and results from inference (ranked genes lists) and evaluation as reported in the manuscript. You may then use the trained model to run inference and evaluation without the need to train them from scratch. If you would like to train all models from scratch, you may skip to the next step.
```bash
bash scripts/download_all_results.sh
```

## Reproduce all results

To reproduce all results in the manuscript, run the following script to run inference and evaluation only and use the saved models without training them from scratch. Note that this assumes you previously downloaded all the saved models and graphs in `results`
```bash
bash scripts/reproduce_all_results.sh
```

If you already have the ranked genes, you may speed this up even further by just running evaluating the saved ranked genes to skip both the training and the inference steps:
```bash
bash scripts/evaluate_all.sh
```
# Usage
In the following steps, we demonstrate how to use the model, including training a model from scratch, using the model for inference by predicting a ranked list of disease genes given one or a list of diseases, and running evaluation on the ranked genes list predicted against known associated disease genes for the given diseases.

## Train a new model
To train a new model, you must have a config.yml file, ideally inside the `experiment_dir` passed as an argument. Refer to `default_config.yml` for the format and parameters needed. The same config file should be used for training, prediction, and evaluation to ensure reproduceability. To train a new DisGeneFormer model based on the configuration, call `train.py` and pass the directory containing the `config.yml` file, as shown below:
```bash
python train.py results/best_model
```

## Evaluate a trained DisGeneFormer model
To run inference and evaluation directly, use the following command with the same config file used in training. 
To use the model to predict a ranked list of disease genes, refer to the next step using `predict_disease_genes.py`.
To run evaluation on existing ranked genes list without the need for any model inference, refer to the step after to use `evaluate.py`. 
```bash
python predict_genes.py results/best_model
```

## Predict the top ranked genes using the model
The model produces a list of ranked genes for each given disease defined in `data/eval_diseases.tsv`. Diseases can be provided either as OMIM IDs or as UMLS CUIs which will each be treated as a set of OMIM IDs and mapped accordingly, based on the mapping defined in `data/test/UMLS_OMIM_map.tsv`. 

To get a list of ranked disease genes for the diseases defined in `data/eval_diseases.tsv`, run `predict_disease_genes.py <path_to_saved_model>` as done below:
```bash
python predict_genes.py results/best_model --predict-only
```

## Evaluate model predictions for DisGeneFormer and other model predictions
To evaluate the ranked genes of a model on the list of evaluation diseases, without requiring training or inference, simply run `python evaluate.py <path_to_ranked_genes>`. This expects a `ranked_genes` folder inside the directory and each ranked genes file should follow the naming convention `<diseaseId>_ranked_genes.tsv`, such as: 

```bash
python evaluate.py results/best_model
```

