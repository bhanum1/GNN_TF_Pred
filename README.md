# Thermal_Fluid_Prediction_GNN
Collection of my work during Summer 2024 for the AI4ChemS Lab at the University of Toronto.


## Installation
First, run pip install git+https://github.com/bhanum1/chemprop.git to install the modified version of Chemprop used in the code found in this repository.

Then create conda environment using the provided environment.yml file.

## Summary of Content

Main_Results contains the best results achieved from training the modified chemprop network on viscosity, vapor pressure, and thermal conductivity data, all of which are found in the Datasets folder.

The model was ran from the command line with the command "chemprop train --config-path [PATH to config file]". Examples of config files for training, hyperparameter optimizing, and transfer learning are included in the Config_Files folder.

The MAML folder contains models, data, and results from the attempted usage of Model Agnostic Meta-Learning (MAML) to improve performance. It also contains a subfolder with a simplified implementation of MAML on a sine regression problem.

Finally, the Barlow_Twins and Report folders were made in collaboration with Hari Om Chadha, and represent our combined efforts to use pretraining and transfer learning to improve performance.

To train a model, first install the environment file in a conda environment, then run train.ipynb in that environment, which will train a default GNN model on the viscosity dataset. Hyperparameters can be changed as well.







