import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from typing import Union


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
    
def plot_with_sd(df: pd.DataFrame, title=None, rolling_window: int = 20):
    measures = df.columns[1:-2]
    n_cols = 2
    n_rows = len(measures) // n_cols
    n_rows += len(measures) % n_cols
    _, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 9))
    for measure, ax in zip(measures, axs.flatten()):
        for exp in df.exp_name.unique():
            data = df.loc[df.exp_name==exp]
            data.iloc[:, :-2] = data.iloc[:,:-2].rolling(rolling_window, min_periods=0).mean()
            data = data.pivot_table(index=data.index, columns="exp_name", values=measure, aggfunc=["mean", "std"])
            ax.plot(data.index, data["mean", exp])
            if "std" in data:
                ax.fill_between(
                    data["mean", exp].index,
                    (data["mean", exp] - data["std", exp]).values.flatten(),
                    (data["mean", exp] + data["std", exp]).values.flatten(),
                    alpha=0.4,
                    interpolate=True
                )
        ax.set(xlabel='Period steps', ylabel=measure.split("/")[1], title=title)
    plt.show()

def main(logdir_or_logfile, data_prefix: Union[str, list]):
    if type(data_prefix) == list:
        df_list = [load_data(logdir_or_logfile, prefix) for prefix in data_prefix]
    else:
        df_list = [load_data(logdir_or_logfile, data_prefix)]
    for df in df_list:
        plot_with_sd(df)

if __name__ == "__main__":
    data_prefix = ["eps_stats", "period_stats"]
    main("C:/Users/s_haas09/VSC_Python/EcoModRL/runs/", data_prefix) #CassKoopmans-v0__casskoopmans_ppo__1__1712927370