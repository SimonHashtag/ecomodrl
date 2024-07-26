import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import glob
from typing import Union


color_map = [
    ("#7ab51d", "#dff4bd"),  # green
    ("#AF0078", "#ffb3e7"),  # violett
    ("#00578a", "#b3e3ff"),  # blue
    ("#333f48", "#d2dadf"),  # dark bluegrey
    ("#009dd1", "#b3ecff"),  # bright blue
    ("#fff300", "#fffa95"),  # yellow
    ("#000000", "#a7a7a7"),  # black
    ("#ff00e8", "#ffb8f9"),  # pink
    ("#00b09a", "#8df7ea"),  # turquoise
    ("ff0000", "#ff9797"),  # red 2
]


def get_color(idx):
    return color_map[idx] if idx <= len(color_map) - 1 else color_map[0]


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

def idxmaxs(df, n=5):
    return list(df.sort_values().head(n).index)

def idxmins(df, n=5):
    return list(df.sort_values().tail(n).index)

def iqm(df: pd.DataFrame, q: float = 0.1):
    """InterQuantileMean: 
    Filter top and worst q-quantile seeds according to mean"""
    drop_n = int(np.ceil(q * df.shape[1]))
    return df.drop(idxmaxs(df.mean(), drop_n)+idxmins(df.mean(), drop_n), axis=1)

def plot_with_sd(df: pd.DataFrame, title=None, rolling_window: int = 1, xmax: int = 10e4, ymin: int = -5, ymax: int = 5):
    measures = df.columns[1:-2]
    n_cols = 2
    n_rows = len(measures) // n_cols
    n_rows += len(measures) % n_cols
    _, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 9))
    for measure, ax in zip(measures, axs.flatten()):
        yaxmin = ymin
        yaxmax = ymax
        for i, exp in enumerate(df.exp_name.unique()):
            color, color_light = get_color(i)
            data = df.loc[df.exp_name==exp]
            data.iloc[:, 1:-2] = data.iloc[:,1:-2].rolling(rolling_window, min_periods=0).mean()
            data = data.pivot(columns="training_seed", values=measure).ffill()
            data = iqm(data, 0.2)
            data = data.apply(pd.DataFrame.describe, 1)[["mean", "std"]]
            data = pd.concat([data,pd.DataFrame(data.iloc[-1:,].values, index=[df.index.max()], columns=data.columns)])
            #data = data.pivot_table(index=data.index, columns="exp_name", values=measure, aggfunc=["mean", "std"])
            #data = data.iloc[1:int((len(data)/2)),]
            ax.plot(data.index, data["mean"], color=color, label=exp.split("_", 1)[1].upper())
            if "std" in data:
                ax.fill_between(
                    data["mean"].index,
                    (data["mean"] - data["std"]).values.flatten(),
                    (data["mean"] + data["std"]).values.flatten(),
                    color=color_light,
                    alpha=0.75,
                    interpolate=True
                )
            if data["mean"].values.min() < yaxmin:
                yaxmin = data["mean"].values.min()
            if data["mean"].values.max() > yaxmax:
                yaxmax = data["mean"].values.max()
        ax.set_xlim([0, df.index.max()])
        ax.set_ylim([yaxmin-1, yaxmax+1])
        ax.hlines(-55.674136089520125, 0, xmax, linestyle='--', color="#333F48")
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 335)
        ax.set(xlabel='Environment steps', ylabel=measure.split("/")[-1].replace("_", " ").title(), title=title)
        ax.legend(loc=4)
    plt.show()

def main(logdir_or_logfile, data_prefix: Union[str, list]):
    if type(data_prefix) == list:
        df_list = [load_data(logdir_or_logfile, prefix) for prefix in data_prefix]
    else:
        df_list = [load_data(logdir_or_logfile, data_prefix)]
    for df in df_list:
        plot_with_sd(df)

if __name__ == "__main__":
    #data_prefix = ["eps_stats", "period_stats"]
    data_prefix = "eval_eps_stats"
    main("C:/Users/s_haas09/VSC_Python/EcoModRL/runs/poster_bernoulli", data_prefix) #CassKoopmans-v0__casskoopmans_ppo__1__1712927370