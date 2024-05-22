# Source: https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py

import glob
import os
import pprint
import traceback
from typing import Optional

import click
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def stats2csv(logs: pd.DataFrame, data_prefix: str, out_dir: str):
    cols = ["step"] + ["walltime"] + [col for col in logs if col.startswith(data_prefix)]
    clean_logs = logs[cols].dropna()
    info = os.path.basename(out_dir).split("__")
    clean_logs["exp_name"] = info[1]
    clean_logs["seed"] = info[2]
    out_file = os.path.join(out_dir, data_prefix + "__" + os.path.basename(out_dir).split("__", 1)[1] + ".csv")
    clean_logs.to_csv(out_file, index=None)
    clean_logs.set_index("step", inplace = True)
    clean_logs.index.name = None
    return clean_logs


def extract_stats(logdir_or_logfile, out_dir, data_prefix: list = ["period_stats"]):
    if os.path.isdir(logdir_or_logfile):
        csv_path = glob.glob(os.path.join(logdir_or_logfile, "*.csv"))[0]
        if out_dir is None:
            out_dir = logdir_or_logfile
    elif os.path.isfile(logdir_or_logfile):
        csv_path = logdir_or_logfile
        if out_dir is None:
            out_dir = os.path.dirname(logdir_or_logfile)
    else:
        raise ValueError(
            f'''No csv-file found! {logdir_or_logfile} 
            either has to be the directory to a csv-file 
            or to a folder containing a csv-file.'''
        )
    all_logs = pd.read_csv(csv_path)
    df_list =  [stats2csv(all_logs, prefix, out_dir) for prefix in data_prefix]
    return df_list


# Extraction function
def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame

    Parameters
    ----------
    path : str
        path to tensorflow log file

    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": [], "walltime": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            walltime = list(map(lambda x: x.wall_time, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step, "walltime": walltime}
            r = pd.DataFrame(r)
            r["walltime"] -= walltime[0]
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data


def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


def tflogs2file(logdir_or_logfile: str, 
                write_pkl: bool = False, 
                write_csv: bool = True, 
                out_dir: Optional[str] = None, 
                wide_format: bool = True, 
                data_prefix: list = ["period_stats"]
                ):
    pp = pprint.PrettyPrinter(indent=4)
    if os.path.isdir(logdir_or_logfile):
        # Get all event* runs from logging_dir subdirectories
        event_paths = glob.glob(os.path.join(logdir_or_logfile, "event*"))
        result_file = os.path.basename(logdir_or_logfile)
        if out_dir is None:
            out_dir = logdir_or_logfile
    elif os.path.isfile(logdir_or_logfile):
        event_paths = [logdir_or_logfile]
        result_file = os.path.basename(os.path.dirname(logdir_or_logfile))
        if out_dir is None:
            out_dir = os.path.dirname(logdir_or_logfile)
    else:
        raise ValueError(
            "input argument {} has to be a file or a directory".format(
                logdir_or_logfile
            )
        )
    # Call & append
    if event_paths:
        pp.pprint("Found tensorflow logs to process:")
        pp.pprint(event_paths)
        all_logs = many_logs2pandas(event_paths)

        os.makedirs(out_dir, exist_ok=True)
        if write_csv:
            print("saving to csv file")
            out_file = os.path.join(out_dir, result_file + ".csv")
            print(out_file)
            if wide_format:
                all_logs = all_logs.pivot_table(index=["step", "walltime"], columns="metric", values="value", aggfunc='mean')
                all_logs.to_csv(out_file)
                extract_stats(logdir_or_logfile=logdir_or_logfile, out_dir=out_dir, data_prefix=data_prefix)
            else:
                all_logs.to_csv(out_file)
        if write_pkl:
            print("saving to pickle file")
            out_file = os.path.join(out_dir, result_file + ".pkl")
            print(out_file)
            all_logs.to_pickle(out_file)
    else:
        print("No event paths have been found.")


@click.command()
@click.argument("logdir-or-logfile")
@click.option(
    "--write-pkl/--no-write-pkl", help="save to pickle file or not", default=False
)
@click.option(
    "--write-csv/--no-write-csv", help="save to csv file or not", default=True
)
@click.option("--out-dir", "-o", help="output directory", default=None)
@click.option("--wide-format/--no-wide-format", help="output in wide format?", default=False)
@click.option("--prefix", help="list of prefixes of stored values, e.g. 'period_stats'", multiple=True, default=["period_stats", "eps_stats", "performance"])
def main(logdir_or_logfile: str, write_pkl: bool, write_csv: bool, out_dir: Optional[str], wide_format: bool, prefix: list):
    """This is a enhanced version of
    https://gist.github.com/ptschandl/ef67bbaa93ec67aba2cab0a7af47700b

    This script exctracts variables from all logs from tensorflow event
    files ("event*"),
    writes them to Pandas and finally stores them a csv-file or
    pickle-file including all (readable) runs of the logging directory.

    Example usage:
    Navigate to file location and type:
        python tf_to_csv.py ../../runs/  --write-csv --no-write-pkl --wide-format --prefix "period_stats" --prefix "eps_stats"
    """
    if os.path.isdir(logdir_or_logfile):
        dirs = os.listdir(logdir_or_logfile)
        for dir in dirs:
            tflogs2file(os.path.join(logdir_or_logfile, dir), write_pkl, write_csv, out_dir, wide_format, prefix)
    else:
        tflogs2file(logdir_or_logfile, write_pkl, write_csv, out_dir, wide_format, prefix)


if __name__ == "__main__":
    main()