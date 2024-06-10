# Thermal_Fluid_Prediction_GNN
Collection of my work during Summer 2024 for the AI4ChemS Lab at the University of Toronto.

See the chemprop_commands.txt file for the commands to put into the CLI to run chemprop networks.

Modified versions of chemprop networks are being used. This framework is from my other repository "chemprop", and can be installed with $ pip install git+https://github.com/bhanum1/chemprop.git.

Config files contain all the specifications needed, but path files will need to be changed based on user's file setup.

To do hyperparameter optimization, run $ pip install hyperopt then $ pip install -U "ray[data, train, tune, serve]"