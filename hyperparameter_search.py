

from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import torch

from main import main as run_model


def main(args):
    print("Hyperparameter tuning started.")
    with open(args.hyperparameter_range, "r") as fp:
        param_ranges = json.load(fp)

    print("Selecting random values from:")
    print(param_ranges)

    config_parent_dir = Path(args.config)
    config_parent_dir.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(args.seed)
    run_num = args.exp_run_start
    for i in range(args.n):
        experiment_run = f"{run_num:03d}"
        print()
        print(f"Training run {i}/{args.n} ({experiment_run}) started...")
        params = get_hyperparameters(param_ranges, rng)
        params["experiment_run"] = experiment_run
        params["push_start"] = params["num_warm_epochs"] + params["push_start_delay"]
        print("Selected parameters: ")
        print(params)
        config_dir = config_parent_dir / experiment_run
        config_dir.mkdir()
        with open(config_dir / "hyperparameters.json", "w") as fp:
            json.dump(params, fp, indent=2)
        run_model(config_dir / "hyperparameters.json", args.workers)
        print(f"Training run {i}/{args.n} completed.")
        run_num += 1

    print("Done!")


def get_hyperparameters(param_ranges, rng):
    params = {}
    for k, v in param_ranges.items():
        params[k] = get_param(v, rng)
    return params


def get_param(values, rng):
    if values is None:
        param = None
    elif type(values) == dict:
        return get_hyperparameters(values, rng)
    elif len(values) == 2:
        if type(values[0]) == float:
            param = rng.random() * (values[1] - values[0]) + values[0]
        elif type(values[0]) == int:
            param = rng.integers(values[0], values[1])
        else:
            param = rng.choice(values)
    else:
        param = rng.choice(values)

    if type(param) == np.ndarray:
        param = param.tolist()
    elif type(param) == np.int64:
        param = int(param)
    elif type(param) == np.float64:
        param = float(param)
    elif type(param) == np.bool:
        param = bool(param)
    elif type(param) == np.bool_:
        param = bool(param)

    return param


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--hyperparameter-range",
        type=str,
        help=".json containing hyperparameters to tune and their min/max",
        default="./hyperparameter_ranges.json"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="No. runs",
        default=1
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for generating the hyperparameters",
        default=None
    )
    parser.add_argument(
        "--exp-run-start",
        type=int,
        help="Experiment number to start on",
        default=100
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="No. workers for dataloading",
        default=4
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Location to save config (hyperparameter) .jsons",
        default="./configs/hyper/"
    )
    args = parser.parse_args()
    main(args)
