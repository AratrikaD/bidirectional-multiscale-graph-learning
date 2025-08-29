import datetime
import os
import sys
import tomli
from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from src.utils.azure_utils import get_hyperparameter_settings, update_command_string, create_inputs
from azure.ai.ml.sweep import Choice, Uniform

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_ml_client(config_path: str) -> MLClient:
    """Summary
    This function returns an MLClient object that can be used to interact with the AzureML workspace locally.

    Args:
        config_path (str): _description_

    Returns:
        MLClient: _description_
    """
    config = read_config(config_path)
    creds = config["credentials"]

    return MLClient(
        DefaultAzureCredential(exclude_shared_token_cache_credential=True),
        creds["subscription_id"],
        creds["resource_group"],
        creds["workspace_name"],
    )


def read_config(path: str) -> dict:
    try:
        with open(path, "rb") as file:
            config = tomli.load(file)
    except:
        print(f"Make sure to correctly specify the config.toml file in {path}")
        raise
    return config

def get_hyperparameter_spaces(hyperparameters: dict) -> dict:
    spaces = {}
    for hp, hp_specs in hyperparameters.items():
        if hp_specs['type'] == 'int':
            spaces[hp] = Choice(values=[integer for integer in range(hp_specs['min'], hp_specs['max'] + 1)])
        elif hp_specs['type'] == 'float':
            spaces[hp] = Uniform(min_value=hp_specs['min'], max_value=hp_specs['max'])
        elif hp_specs['type'] == 'category':
            spaces[hp] = Choice(values=hp_specs['options'])
    return spaces


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# from QLearner.azure_utils.core import get_ml_client

if __name__ == "__main__":
    ### Note: First do `az login' in the terminal to authenticate with Azure
    ### Note 2: Make sure to have the correct config.toml file in the specified path with the correct credentials

    from configs import DirectoryConfig as DIR

    # Set up Azure ML client
    config_path = os.path.join(DIR.AZUREML, "config.toml")
    ml_client = get_ml_client(config_path)
    hyperparameter_path = os.path.join(DIR.ROOT, 'hyperparameter_settings.toml')
    hp_settings = get_hyperparameter_settings(hyperparameter_path)
    hyperparameters, experiment_settings = hp_settings['hyperparameters'], hp_settings['experiment']
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    experiment_name = f"version_{timestamp}"


    
    # command_string = "python dynamic_baseline.py "
    # command_string = update_command_string(command_string, hyperparameters)

    # Prepare inputs that will be passed as arguments to the command
    hyperparameter_inputs = create_inputs(hyperparameters)
    config = read_config(config_path)
    env = config["environment"]
    data_path = os.path.join(env["data_path"], "housing-data")
    hyperparameter_inputs['input_data'] = Input(type=AssetTypes.URI_FOLDER, path=data_path)

    
    # Use pre-created environment
    env_name = "azureml:CUDAML3:2"

    # Set compute target to LowPriorityGpuCluster
    # compute_name = "DiffusionPriorityGPUCluster"
    compute_name ="PriorityGraphGPUCompute"
    # compute_name = "PriorityGraphDataCompute"

    data_type = AssetTypes.URI_FOLDER
    # mode = InputOutputModes.DOWNLOAD
    
    
    identity = None
    print(hyperparameter_inputs)

    command_string = "python main.py "

    command_string = update_command_string(command_string, hyperparameters)
    # Define the command job
    command_job = command(
        code="gnns-for-property-valuation", ## Adjust this path based on your project structure
        command=command_string, ## Adjust this path based on your project structure
        inputs=hyperparameter_inputs,
        compute=compute_name,
        environment=env_name,
        outputs={},
        name="hyperparameter_tuning_w_2", # Name of the job
        experiment_name=experiment_name,
        identity=identity
    )
    # command_job_for_sweep = command_job(**get_hyperparameter_spaces(hyperparameters))
    
    # print(command_job_for_sweep)
    # # apply the sweep parameter to obtain the sweep_job
    # sweep_job = command_job_for_sweep.sweep(
    #     compute= compute_name,
    #     sampling_algorithm=experiment_settings['sampling_algorithm'],
    #     primary_metric=experiment_settings['primary_metric'],
    #     goal=experiment_settings['goal'],
    # )

    # # define the limits for this sweep
    # sweep_job.set_limits(max_total_trials=experiment_settings['max_total_trials'], 
    #                      max_concurrent_trials=experiment_settings['max_concurrent_trials'])

    # # submit the sweep
    # returned_sweep_job = ml_client.create_or_update(sweep_job)
    # # Submit the job
    ml_client.jobs.create_or_update(command_job)
    print(f"Job {experiment_name} submitted to Azure ML.")
