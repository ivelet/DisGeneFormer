# DisGeneFormer
An Attention-based Graph Transformer for Disease Gene Prioritization. 

We describe our approach in the Preprint [“DisGeneFormer: Precise Disease Gene Prioritization by Integrating Local and Global Graph Attention”](https://www.biorxiv.org/content/10.64898/2026.03.11.711106).

Any publication that discloses findings arising from using this source code, the model parameters, or outputs produced by those should also cite this paper.

If you encounter any problems, please open an Issue on this repository.

## Table of contents
* [Setup](#setup)
* [Download all datasets](#download-all-datasets)
* [Reproduce all experiments](#reproduce-all-experiments)
* [Download all models and results](#download-all-datasets)
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

## Download all datasets
To download all raw datasets needed to reproduce all experiments in the manuscript, use:
```bash
bash scripts/download_all_datasets.sh
```

To run the Model Comparison experiment, the ranked genes lists reported by the other methods must be downloaded using:
```bash
bash scripts/download_model_comparison_results.sh
```

## Reproduce all experiments
To reproduce all experiments from scratch in the manuscript, you may run the following script. Note that this includes training each model from scratch and then running inference and evaluation which may take some time. 

<!-- NOTE: there is a total of 15-25 experiments included to reproduce all results in the manuscript which will take ~ -->

To save computational time, skip to the next step to download all models and results to avoid having to run all training and inference from scratch.
```bash
bash scripts/reproduce_all_experiments.sh
```

## Download all models and results
To avoid having to retrain all models from scratch to reproduce all experiments, we additionally provide all trained models and results from inference (ranked genes lists) and evaluation as reported in the manuscript. You may then use the trained model to run inference and evaluation without the need to train them from scratch.

Note that all reported results are the average over 5 training folds and thus all saved models from 5 folds are required to reproduce the results. Downloading all models will be approximately 30 GB. To save storage space, you may instead skip this step and download all results including the ranked genes list in the next step which excludes all the saved models. This approximately reduces the storage requirements from 30 GB to 500 MB.
```bash
bash scripts/download_all_models_and_results.sh
```

To download only all the results including ranked genes lists without model files, run the following:
```bash
bash scripts/download_all_results.sh
```

## Reproduce all results

To reproduce all results in the manuscript WITHOUT training any models, run the following script to run inference and evaluation only and use the saved models without training them from scratch. Note that this assumes you previously downloaded all the saved models and graphs in `results`
```bash
bash scripts/reproduce_all_results.sh
```

If you already have the ranked genes, you may speed this up even further by just running evaluating the saved ranked genes to skip both the training and the inference steps:
```bash
bash scripts/evaluate_all.sh
```
## Usage
In the following steps, we demonstrate how to use the model, including training a model from scratch, using the model for inference by predicting a ranked list of disease genes given one or a list of diseases, and running evaluation on the ranked genes list predicted against known associated disease genes for the given diseases.

### Train a new model
To train a new model, you must have a config.yml file, ideally inside the `experiment_dir` passed as an argument. Refer to `default_config.yml` for the format and parameters needed. The same config file should be used for training, prediction, and evaluation to ensure reproduceability. To train a new DisGeneFormer model based on the configuration, call `train.py` and pass the directory containing the `config.yml` file, as shown below:
```bash
python train.py results/DisGeneFormer 
```
Or to train the version of DisGeneFormer with filtered edges 
```bash
python train.py results/DisGeneFormer_filtered
```

### Evaluate a trained DisGeneFormer model
To run inference and evaluation directly, use the following command with the same config file used in training. 
To use the model to predict a ranked list of disease genes, refer to the next step using the same script with the`--predict-only` flag. Note that this method runs inference and then evaluates on all 5 folds of the trained model and then averages over them as was reported for all experiments in the manuscript. 
To run evaluation on existing ranked genes list without the need for any model inference, refer to the step after to use `evaluate.py`. 
```bash
python predict_genes_fold.py results/DisGeneFormer
```
Or the filtered version
```bash
python predict_genes_fold.py results/DisGeneFormer_filtered
```

### Predict the top ranked genes using the model
The model produces a list of ranked genes for each given disease defined in `data/eval_diseases.tsv`. Diseases can be provided either as OMIM IDs or as UMLS CUIs which will each be treated as a set of OMIM IDs and mapped accordingly, based on the mapping defined in `data/test/UMLS_OMIM_map.tsv`. 

To get a list of ranked disease genes for the diseases defined in `data/eval_diseases.tsv`, run `predict_disease_genes.py <path_to_saved_model>` as done below:
```bash
python predict_genes_fold.py results/DisGeneFormer --predict-only
```
Or the filtered version
```bash
python predict_genes_fold.py results/DisGeneFormer_filtered --predict-only
```

### Evaluate model predictions for DisGeneFormer and other models
To evaluate the ranked genes of a model on the list of evaluation diseases, without requiring training or inference, simply run `python evaluate.py <path_to_ranked_genes>`. This expects a `ranked_genes` folder inside the directory and each ranked genes file should follow the naming convention `<diseaseId>_ranked_genes.tsv` as shown below. To evaluate the average over all model folds as reported in the manuscript, run `evaluate_fold.py` as shown below on the best version of DisGeneFormer: 

```bash
python evaluate_fold.py results/DisGeneFormer
```
Or the filered version
```bash
python evaluate_fold.py results/DisGeneFormer_filtered
```

## Plot results from the manuscript

The following can be used to reproduce all plots from the manuscript.

### Plot model comparison true positive (TP) curves 

First, combine the results for the model comparison to get a table of the top K {5, 20, 50} precision of DisGeneFormer (both the best version and the one trained on the full XC_V3 version of HumanNet) against other DGP methods compared against in the manuscript which are saved in `results/model_comparison` using the following script which can be used to combine any of the results into a single table. The following uses all default values which can be set with flags.

```bash
python plots/scripts/combine_results.py results/model_comparison 
```

Run the following script to plot the number of true positives for each value of K within a range, showing each method as a separate curve and creating a separate plot for each disease to compare different methods on the same disease.

```bash
python plots/scripts/plot_tp_curves.py results/model_comparison --output-dir plots/results/method_comparison_tp_curves --method-names plots/results/method_comparison_tp_curves/method_names.json
```

### Plot HumanNet comparison TP curves
Using the same script, we can plot the results comparing different versions of HumanNet that DisGeneFormer was trained on.

```bash
python plots/scripts/plot_tp_curves.py results/humannet_comparison --output-dir plots/results/humannet_comparison_tp_curves --method-names plots/results/humannet_comparison_tp_curves/method_names.json 
```

### Plot Hard Negatives identity scatter plot
We then plot the identity scatter plot comparing the difference in performance when training on Hard Negatives (HNs) compared to training on randomly generated negative association data (RNs).

```bash
python plots/scripts/plot_identity_scatter.py results/negative_comparison random_negatives hard_negatives --output-dir plots/results/negatives_comparison_identity_scatter --k-value 20 --metric omim_prec --exclude-diseases C0376358 C0009402
```

### Reproduce graph feature ablation table
To reproduce the results in the manuscript observing the effects of removing individual feature types from the gene and disease graphs, including the table of results, run the following:

```bash
bash scripts/run_feature_removal_experiment.sh
```

