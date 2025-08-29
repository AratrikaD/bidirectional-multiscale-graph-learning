import tomli
import argparse



def get_hyperparameter_settings(path: str) -> dict:
    try:
        with open(path, 'rb') as file:
            settings = tomli.load(file)
    except:
        print(f'Make sure {path} points to a correctly specified hyperparameter .toml file.')
        raise
    return settings


def update_command_string(command_string: str, hyperparameters: dict) -> str:
    for hp_name in hyperparameters.keys():
        command_string += f"--{hp_name} " + "${{inputs." + f"{hp_name}" + "}} "
    command_string += "--data_path " + "${{inputs.input_data}}"
    return command_string


def create_inputs(hyperparameters: dict) -> dict:
    inputs = {}
    for hp_name, hp_specs in hyperparameters.items():
        inputs[hp_name] = hp_specs['default']
    return inputs


def parse_hyperparameter_args(hyperparameter_path: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    hyperparameters = get_hyperparameter_settings(hyperparameter_path)['hyperparameters']

    for hp_name, hp_specs in hyperparameters.items():
        if hp_specs.get('array'):
            parser.add_argument(f'--{hp_name}', 
                                nargs='+', 
                                default=hp_specs['default'],
                                help=hp_specs['description'])
        else:
            parser.add_argument(f'--{hp_name}', 
                                default=hp_specs['default'], 
                                help=hp_specs['description'])
    parser.add_argument('--data_path', default='gnns-for-property-valuation\housing-data', help='Path to the data folder')
    return parser.parse_args()

