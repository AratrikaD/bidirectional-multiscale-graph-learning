import sys
import os
import tomli
from src.experiments.global_graph_embedding import run_experiment
from src.utils.azure_utils import parse_hyperparameter_args, get_hyperparameter_settings
# from src.experiments.lightgbm_baseline import run_lgbm_experiment
from src.experiments.final_alt import run_exp
from src.experiments.online_learning import run_exp_online
from src.experiments.final_model_exp import run_final_exp

# from src.experiments.dynamic_baseline import run_exp
import argparse
import mlflow
import time

DIR = os.path.dirname(os.path.realpath(__file__))

def read_config(path: str) -> dict:
    try:
        with open(path, "rb") as file:
            config = tomli.load(file)
    except:
        print(f"Make sure to correctly specify the config.toml file in {path}")
        raise
    return config


# def main(params):
#     script_dir = os.path.dirname(os.path.realpath(__file__))
#     sys.path.append(script_dir)
#     if params.data_path != None:
#         data_path = params.data_path
#     else:
#         data_path = os.path.join(script_dir,"housing-data")
#     print(data_path)
#     config_path = os.path.join(script_dir, "config")

#     hyperparameters = {
#         "seed" : int(params.seed),
#         "batch_size" : int(params.batch_size),
#         "learning_rate" : float(params.learning_rate),
#         "gnn_hidden_channels" : int(params.gnn_hidden_channels),
#         "nn_hidden_channels" : int(params.nn_hidden_channels),
#     }
#     mlflow.log_params(hyperparameters)
    
#     run_experiment(data_path, config_path, hyperparameters)


# if __name__ == "__main__":
#     mlflow.create_experiment(str(time.time()))
#     with mlflow.start_run():
#         HYPERPARAMETER_PATH = os.path.join(
#                 DIR, "hyperparameter_settings.toml"
#             )
#         params = parse_hyperparameter_args(HYPERPARAMETER_PATH)
#         print(f"{params}")
#         main(params)
def main(params):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(script_dir)
        print(params.data_path)
        if params.data_path != None:
            data_path = params.data_path
            # output_path = os.path.join(script_dir, "outputs")
        else:
            data_path = os.path.join(script_dir,"housing-data")
            # output_path = os.path.join(script_dir, "outputs")
        print(data_path)
        run_final_exp(data_path, hyperparameters=params)
        # run_exp_online(data_path)
        # run_lgbm_experiment(data_path)

if __name__ == "__main__":
    mlflow.create_experiment(str(time.time()))
    with mlflow.start_run():
        DIR = os.path.dirname(os.path.realpath(__file__))
        HYPERPARAMETER_PATH = os.path.join(
                DIR, "hyperparameter_settings.toml"
            )
        params = parse_hyperparameter_args(HYPERPARAMETER_PATH)
        print(f"{params}")
        main(params)