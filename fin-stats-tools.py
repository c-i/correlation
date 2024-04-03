import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from datetime import date as d
import time
import argparse
import os




DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
HEADER = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]




def get_args():
    parser = argparse.ArgumentParser(description="Statistics tools for finance.")
    subparser = parser.add_subparsers(dest="stats_tool", required=True, help="stats tool")

    correlation_parser = subparser.add_parser("correlation", help="Finds Pearson correlations between log returns of given assets.")

    correlation_parser.add_argument(
        "assets",
        metavar="assets",
        help="USDT denominated trading pairs.  e.g. BTCUSDT ETHUSDT (if {all} is provided as the first argument all assets will be used)",
        nargs="+"
    )
    correlation_parser.add_argument(
        "--start",
        help="(required) Start date in iso format.  e.g. 2020-12-30",
        required=True
    )
    correlation_parser.add_argument(
        "--end",
        help="(required) End date in iso format (up to the end of last month).  e.g. 2021-12-30",
        required=True
    )
    correlation_parser.add_argument(
        "-p",
        help="(bool) Choose whether to plot correlation matrix heatmap of top n assets.",
        action="store_true",
        default=False
    )
    correlation_parser.add_argument(
        "-n",
        help="(int) Number of assets to plot in correlation matrix heatmap.  Default: 10",
        type=int,
        default=10
    )
    correlation_parser.add_argument(
        "-m",
        help="(bool) Save correlation matrix to csv file.",
        action="store_true",
        default=False
    )
    correlation_parser.add_argument(
        "-l",
        help="(bool) Save list of asset pair correlations sorted from greatest to least to csv file.",
        action="store_true",
        default=False
    )
    correlation_parser.add_argument(
        "--interval",
        help="(int) Interval for which to calculate returns as a multiple of granularity. e.g. 1 (an interval of 1 with granularity 1d would calculate returns once per day).  Default: 1",
        type=int,
        default=1
    )
    correlation_parser.add_argument(
        "--granularity",
        help="Granularity of k-line data.  e.g. 1d (default: 1d)",
        default="1d"
    )
    correlation_parser.add_argument(
        "--data_dir",
        help=f"Directory where k-line data is stored.  Default: {DIR}/spot/monthly/klines",
        default=f"{DIR}/spot/monthly/klines"
    )
    correlation_parser.add_argument(
        "--component",
        help="CSV header label used to calculate log returns.  Default: close",
        default="close"
    )
    correlation_parser.add_argument(
        "--index",
        help="CSV header label used to retrieve timestamps.  Default: close_time",
        default="close_time"
    )

    return parser.parse_args()



# takes trading pair (string), directory (string), granularity (string) and returns dictionary of dataframes of the historical data of those pairs: e.g. {"BTCUSDT": df,"ETHUSDT": df}
def to_dfs(assets_arg, dir, granularity):
    data_lists = {}
    dfs = {}
    
    if assets_arg[0] == "all":
        assets = os.listdir(dir)
    else:
        assets = assets_arg
        
    print("Reading CSVs")
    for asset in assets:
        path = f"{dir}/{asset}/{granularity}"

        csv_files = os.listdir(path)

        data_lists[asset] = []

        for file in csv_files:
            with open(path + "/" + file, newline="") as csvfile:
                reader = csv.reader(csvfile)

                if next(reader)[0].replace(".", "").isnumeric():
                    df = pd.read_csv(f"{path}/{file}", names=HEADER)
                    data_lists[asset].append(df)
                
                else:
                    df = pd.read_csv(f"{path}/{file}")
                    data_lists[asset].append(df)

        if len(data_lists[asset]) == 0:
            continue

        dfs[asset] = pd.concat(data_lists[asset], ignore_index=True)

    return dfs


# takes dictionary of dataframes (converted from csvs) and constructs one dataframe according to one column of the csvs and crops to desired timeframe
# excludes data for assets that dont exist for the entire time frame
def combine_by_component(df_dict, start, end, component="close", index="close_time",):
    series_dict = {}
    # converting to dictionary of series in case dates are different between datasets (eg missing dates)
    for asset in df_dict:
        s = df_dict[asset][component]
        timestamps = df_dict[asset][index]

        dates = []
        for timestamp in timestamps:
            date = d.fromtimestamp(timestamp / 1000)
            dates.append(date.isoformat())

        
        if start in dates and end in dates:
            # catch data with duplicate dates, usually indicative of delistings:
            i = 1
            duplicate = False
            while i < len(dates):
                if dates[i] == dates[i - 1]:
                    print("Duplicate dates, unable to process: ", asset)
                    duplicate = True
                i += 1
            if duplicate:
                continue

            s.index = dates

            series_dict[asset] = s[start:end]
            # print("Added: ", asset)
    
    df = pd.DataFrame(series_dict)

    return df



# takes trading pair price dataframes and interval and returns correlation matrix of log returns
def log_returns_corr(price_df, interval=1):
    log_r_df = np.log(price_df) - np.log(price_df.shift(interval))
    corr_matrix = log_r_df.corr(method="pearson")

    corr_matrix = corr_matrix.dropna(axis=0, how="all")
    corr_matrix = corr_matrix.dropna(axis=1, how="all")

    return corr_matrix



# takes correlation matrix and returns series of highest correlations indexed by correlated asset pairs
def corr_series(corr_matrix):
    corr_pair_list = []
    rho_list = []
    # indices = corr_matrix.index
    columns = corr_matrix.columns

    start_col = 1
    for index, row in corr_matrix.iterrows():
        pairs = [f"{index}-{column}" for column in columns[start_col:]]
        
        corr_pair_list.extend(pairs)
        rho_list.extend(row.iloc[start_col:])

        start_col += 1

    corr_series = pd.Series(data=rho_list, index=corr_pair_list).sort_values(ascending=False)

    return corr_series



# returns list of top n assets with highest correlations
def top_assets(corr_series, n=10):
    indices = corr_series[:int(n/2)].index
    assets = []

    for index in indices:
        assets.extend(index.split("-"))
    
    return assets




def plot_heatmap(corr_matrix):
    sns.heatmap(corr_matrix, annot=True, cmap="viridis")
    plt.show()




class Correlation:
    def __init__(self, assets, data_dir, granularity, start, end, interval, n, component, index):
        self.df_dict = to_dfs(assets, data_dir, granularity)
        self.price_df = combine_by_component(self.df_dict, start, end, component=component, index=index)
        self.corr_matrix = log_returns_corr(self.price_df, interval)
        self.corr_s = corr_series(self.corr_matrix)
        self.top_assets = top_assets(self.corr_s, n)
        
        sub_df_dict = to_dfs(self.top_assets, data_dir, granularity)
        sub_price_df = combine_by_component(sub_df_dict, start, end)
        self.corr_submatrix = log_returns_corr(sub_price_df, interval)




def main():
    args = get_args() 

    corr = Correlation(args.assets, args.data_dir, args.granularity, args.start, args.end, args.interval, args.n, args.component, args.index)


    print(corr.corr_submatrix)


    if not os.path.isdir("output"):
        os.mkdir("output")

    if args.m:
        corr.corr_matrix.to_csv(f"{DIR}/output/corr-matrix-{args.start}-to-{args.end}.csv", compression=None)
        print(f"saved csv to {DIR}/output/corr-matrix-{args.start}-to-{args.end}.csv")

    if args.l:
        corr.corr_s.to_csv(f"{DIR}/output/corr-list-{args.start}-to-{args.end}.csv", compression=None)
        print(f"saved csv to {DIR}/output/corr-list-{args.start}-to-{args.end}.csv")
    
    if args.p:
        plot_heatmap(corr.corr_submatrix)


if __name__ == "__main__":
    main()




# TODO: implement async file io (way too much effort, pandas read_csv is not asynchronous)