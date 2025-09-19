import os
import glob
import pandas as pd
import numpy as np

from typing import Union
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


def load_data(logdir_or_logfile: str, data_prefix: str):
    if os.path.isdir(logdir_or_logfile):
        dirs = os.listdir(logdir_or_logfile)
        csv_path = [glob.glob(os.path.join(logdir_or_logfile, dir, data_prefix + "*.csv"))[0] for dir in dirs]
        df = pd.concat([pd.read_csv(p) for p in csv_path])
    elif os.path.isfile(logdir_or_logfile):
        csv_path = logdir_or_logfile
        pd.read_csv(csv_path)
    else:
        raise ValueError(
            f'''No csv-file found! {logdir_or_logfile} 
            either has to be the directory to a csv-file 
            or to a folder containing a csv-file.'''
        )
    df.set_index("step", inplace = True)
    df.index.name = None
    return df

def prepare_data(dataframe, df_label="eps_stats", score_label="episodic_return"):
    data = dataframe[df_label]
    exp_names = pd.unique(data["exp_name"])
    algos = [x.split("_")[1].upper() for x in exp_names]
    grouped_dataframe = data.loc[:, ["exp_name", "seed", f"{df_label}/{score_label}"]].groupby("exp_name")
    score_dict = dict()
    for algo, exp in zip(algos, exp_names):
        # select experiment, pivot dataframe and fill na values (due to early stopping) with forward fill
        wide_df = grouped_dataframe.get_group(exp).pivot(columns="seed", values=f"{df_label}/{score_label}").ffill()
        score_array = np.expand_dims(wide_df.transpose(), 1)
        #score_array = wide_df.transpose().to_numpy()
        score_dict[algo] = score_array
    return score_dict

def main(logdir_or_logfile, data_prefix: Union[str, list]):
    data_prefix = [data_prefix] if isinstance(data_prefix, str) else data_prefix
    df_dict = {prefix: load_data(logdir_or_logfile, prefix) for prefix in data_prefix}
    return df_dict
    # for df in df_list:
    #     plot_with_sd(df)

if __name__ == "__main__":
    data_prefix = ["eps_stats", "period_stats"]
    df_dict = main("C:/Users/s_haas09/VSC_Python/EcoModRL/runs/test", data_prefix)
    score_dict = prepare_data(df_dict)
    #score_dict["PPO"] = score_dict["PPO"][~np.isnan(score_dict["PPO"])]
    aggregate_func = lambda x: np.array([
        metrics.aggregate_median(x),
        #metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x),
        #metrics.aggregate_optimality_gap(x)
        ])
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=50000)
    print("Finished")