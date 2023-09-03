import pandas as pd
import numpy as np
import glob
import os
import json
import argparse
import matplotlib.pyplot as plt


def parse_trial(trial_directory):
    metric_file = os.path.join(trial_directory, '.nni', 'metrics')
    parameter_file = os.path.join(trial_directory, 'parameter.cfg')

    with open(metric_file) as f:
        # Only read last line from the metrics file, which should contain the final results
        # Also strip leading characters
        metric = json.loads(f.readlines()[-1][8:])
        metric['value'] = json.loads(metric['value'])

    with open(parameter_file) as f:
        parameter = json.loads(f.read())

    assert metric['parameter_id'] == parameter['parameter_id']

    return {
        'parameter_id': metric['parameter_id'],
        **metric['value'],
        **parameter['parameters']
    }


def parse_trials(run_directory):
    trials = glob.glob(os.path.join(run_directory, 'trials', '*'))
    print(f'Found {len(trials)} in {run_directory}')
    results = []

    for t in trials:

        try:
            results.append(parse_trial(t))
        except Exception as e:
            print(f'Error in trial {t}:\n {e}')
            continue

    return pd.DataFrame(results)


def plot_top(data, metric='default', ignore_columns=None, out_dir=None):
    if ignore_columns is None:
        ignore_columns = []

    data = data.sort_values(by=metric, ascending=True)

    keys = [k for k in data.keys() if k not in [metric, 'parameter_id'] + ignore_columns]
    for k in keys:
        try:
            fig, ax = plt.subplots(figsize=(15, 5))
            ax.set_title(k)
            column_type = type(data[k][0])

            quartiles = [0.01, 0.05, .1, .2, .3, .4, .5]
            bar_width = 1.0 / len(quartiles)

            for i, q in enumerate(quartiles):
                quant = data[metric].quantile(q)
                d = data[data[metric] <= quant][k]
                label = f'q{int(q * 100):02}'

                if np.issubdtype(column_type, np.number):
                    _, _, _ = ax.hist(d, bins=100, label=label,
                                      density=True, cumulative=True, histtype='step')
                elif np.issubdtype(column_type, np.str) or np.issubdtype(column_type, np.bool):
                    hist = {
                        **{k: 0 for k in set(data[k])},  # All keys
                        **{x: list(d).count(x) / d.count() for x in set(d)}
                    }
                    keys = hist.keys()
                    values = hist.values()
                    x = np.arange(len(keys))
                    ax.bar(x + i * bar_width, values, width=bar_width, label=label)
                    ax.set_xticks(x + bar_width * len(quartiles) / 2)
                    ax.set_xticklabels(keys)
                else:
                    print(f'Unknown column type {column_type} in column {k}')
            ax.legend()
            ax.grid()

            if out_dir:
                fig.savefig(os.path.join(out_dir, f'{k}.pdf'))
            else:
                fig.show()
        except Exception as e:
            print(f'Exception during plotting of {k}: \n  {e}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", type=str, help="Set to the directory containing the nni results.")
    p.add_argument("--csv", type=str, help="Set to filename, imported data is written then in CSV format.")
    p.add_argument("--plot", type=str, help="Set to directory name. If set plots are generated and stored in the given directory")
    p.add_argument("--metric", type=str, default='default', help="Metric to use to compute the quartiles for plotting. Default: 'default'")
    args = p.parse_args()

    df = parse_trials(args.data)

    if args.csv:
        print(f'Writing result to {args.csv}')
        df.to_csv(path_or_buf=args.csv, sep=';', index=False)

    if args.plot:
        print('Generating plots')
        plot_top(df, out_dir=args.plot, metric=args.metric, ignore_columns=[])


if __name__ == "__main__":
    main()
