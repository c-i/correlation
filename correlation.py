import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from datetime import datetime as dt
from datetime import date as d
from statsmodels import tsa
from statsmodels.tsa.vector_ar import vecm
import time
import argparse
import os




DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
HEADER = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]




def get_args():
    parser = argparse.ArgumentParser(description="Finds Pearson correlation between log returns of given assets.")

    parser.add_argument(
        "assets",
        metavar="assets",
        help="USDT denominated trading pairs.  e.g. BTCUSDT ETHUSDT (if {all} is provided as the first argument all assets will be used)",
        nargs="+"
    )
    parser.add_argument(
        "-start",
        metavar="",
        help="(required) Start date in iso format.  e.g. 2020-12-30",
        required=True
    )
    parser.add_argument(
        "-end",
        metavar="",
        help="(required) End date in iso format (up to the end of last month).  e.g. 2021-12-30",
        required=True
    )
    parser.add_argument(
        "-plot",
        metavar="",
        help="(bool) Choose whether to plot correlation matrix heatmap.  Default: False.  Always False if {all} argument provided for assets.",
        type=bool,
        default=False
    )
    parser.add_argument(
        "-csv_m",
        metavar="",
        help="(bool) Save correlation matrix to csv file.  Default: False.",
        type=bool,
        default=False
    )
    parser.add_argument(
        "-csv_l",
        metavar="",
        help="(bool) Save list of asset pair correlations sorted from greatest to least to csv file.  Default: False",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--interval",
        metavar="",
        help="(int) Interval for which to calculate returns as a multiple of granularity. e.g. 1 (an interval of 1 with granularity 1d would calculate returns once per day).  Default: 1",
        type=int,
        default=1
    )
    parser.add_argument(
        "--granularity",
        metavar="",
        help="Granularity of k-line data.  e.g. 1d (default: 1d)",
        default="1d"
    )
    parser.add_argument(
        "--data_dir",
        metavar="",
        help=f"Directory where k-line data is stored.  Default: {DIR}/spot/monthly/klines",
        default=f"{DIR}/spot/monthly/klines"
    )

    # return parser.parse_args(["all", "-start", "2020-09-01", "-end", "2021-09-01"])
    # , "-csv", "True"
    return parser.parse_args(["BTCUSDT", "ETHUSDT", "LINKUSDT", "AAVEUSDT", "MATICUSDT", "AVAXUSDT", "SOLUSDT", "DYDXUSDT", "UNIUSDT", "-start", "2022-04-01", "-end", "2023-04-01", "-plot", "True"])



# takes trading pair (string), directory (string), granularity (string) and returns dictionary of dataframes of the historical data of those pairs: e.g. {"BTCUSDT": df,"ETHUSDT": df}
def to_dfs(assets_arg, dir, granularity):
    data_lists = {}
    dfs = {}

    if assets_arg[0] == "all":
        assets = os.listdir(dir)
    else:
        assets = assets_arg
        

    for asset in assets:
        path = f"{dir}/{asset}/{granularity}"
        try:
            csv_files = os.listdir(path)
        except FileNotFoundError:
            raise

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




def crop_period(price_df_arg, start, end):
    timestamps = np.array(price_df_arg.index)

    dates = []
    for timestamp in timestamps:
        date = d.fromtimestamp(timestamp / 1000)
        dates.append(date.isoformat())

    price_df = price_df_arg
    price_df.index = dates

    return price_df[start:end]




def combine_by_component(df_dict, component="close", index="close_time"):
    series_dict = {}
    # converting to dictionary of series in case dates are different between datasets (eg missing dates)
    for asset in df_dict:
        s = df_dict[asset][component]
        if index != None:
            s.index = df_dict[asset][index]

        series_dict[asset] = s

    df = pd.DataFrame(series_dict)

    return df



# takes trading pair price dataframes and interval and returns correlation matrix of log returns
def log_returns_corr(price_df, interval=1):
    log_r_df = np.log(price_df) - np.log(price_df.shift(interval))
    corr_matrix = log_r_df.corr(method="pearson")

    corr_matrix = corr_matrix.dropna(axis=0, how="all")
    corr_matrix = corr_matrix.dropna(axis=1, how="all")

    return corr_matrix




def corr_series(corr_matrix):
    corr_pair_list = []
    rho_list = []
    indices = corr_matrix.index
    columns = corr_matrix.columns

    start_col = 0
    for index, row in corr_matrix.iterrows():
        pairs = [f"{indices[index]}-{column}" for column in columns[start_col:]]
        corr_pair_list.extend(pairs)

        rho_list.extend(row.iloc[start_col:])

    corr_series = pd.Series(data=rho_list, index=corr_pair_list).sort_values(ascending=False)

    return corr_series




def plot_heatmap(corr_matrix):
    sns.heatmap(corr_matrix, annot=True, cmap="viridis")
    plt.show()




def main():
    print(DIR)
    args = get_args() 
    
    df_dict = to_dfs(args.assets, args.data_dir, args.granularity)
    price_df = crop_period(combine_by_component(df_dict), args.start, args.end)
    corr_matrix = log_returns_corr(price_df, args.interval)
    
    print(corr_matrix)

    df = corr_matrix.copy()
    print(np.fill_diagonal(df.values, 0))
    print(df.idxmax(axis="columns"))

    if args.plot and args.assets[0] != "all":
        plot_heatmap(corr_matrix)

    if args.csv:
        corr_matrix.to_csv(f"{DIR}/out/correlations-all-{time.time()}.csv", compression=None)
        print(f"saved csv to {DIR}/out/correlations-all-{time.time()}.csv")
        
    
   
if __name__ == "__main__":
    main()




# TODO: implement async file io
# trim down "all" corr matrix to just assets with highest correlations in their rows/cols