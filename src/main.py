import numpy as np
import os
import json
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import REGISTRY as run_REGISTRY

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger('INFO')

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    
    # run
    if "use_per" in _config and _config["use_per"]:
        run_REGISTRY['per_run'](_run, config, _log)
    else:
        run_REGISTRY[_config['run']](_run, config, _log)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def _reload_config(config_path):
    try:
        f = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path, "config.json"), "r")
    except:
        f = open(os.path.join(config_path, "config.json"), "r")
        
    try:
        config_dict = json.load(f)
    except:
        raise ValueError('Failed to load config from {}'.format(config_path))

    f.close()

    return config_dict

def recursive_dict_update(d, u):
    from collections import Mapping
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    exp_config = None
    for _i, _v in enumerate(params):
        if "exp" in _v:
            checkpoint_path = _v.split("=")[1]
            exp_config = _reload_config(checkpoint_path)
            config_dict = recursive_dict_update(config_dict, exp_config)
            config_dict["checkpoint_path"] = checkpoint_path
            del params[_i]
            break

    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    if env_config is not None:
        config_dict = recursive_dict_update(config_dict, env_config)
    if alg_config is not None:
        config_dict = recursive_dict_update(config_dict, alg_config)

    # TODO: remove in the future version
    for _i, _v in enumerate(params):
        if "vis_process" in _v:
            config_dict["vis_process"] = True
            break
    if "vis_process" not in config_dict.keys():
        config_dict["vis_process"] = False

    try:
        ma_algo_name = parse_command(params, "mixer", config_dict['mixer'])
        url_algo_name = parse_command(params, "url_algo", config_dict['url_algo'])
        map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
        n_agents = parse_command(params, "env_args.n_agents", config_dict['env_args']['n_agents'])
        config_dict["name"] = url_algo_name + "_" + ma_algo_name + "_" + map_name + "_agents-" + str(n_agents)
    except:
        pass

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    algo_name = parse_command(params, "name", config_dict['name']) 
    file_obs_path = join(results_path, "sacred", map_name, algo_name)

    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver(file_obs_path))

    ex.run_commandline(params)

    # flush
    sys.stdout.flush()
