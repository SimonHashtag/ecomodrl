import os
from json import load
import logging

def import_tuned_hp(setup_args, exp_name: str):
    args = setup_args
    tuned_args_path = os.path.abspath(f"../tune_results/{exp_name}/incumbent.json")
    if os.path.exists(tuned_args_path):
        with open(tuned_args_path) as f:
            params = load(f).get("config")
        for k,v in params.items():
            setattr(args, k, v)
    else: 
        logging.warning(f"Tune path {tuned_args_path} does not exists. Continue with setup args.")
    return args