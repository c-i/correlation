import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from statsmodels import tsa
from statsmodels.tsa.vector_ar import vecm
from datetime import datetime as dt
import time
import argparse
import os




DIR = os.getcwd().replace("\\", "/")
HEADER = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]




def get_args():
    parser = argparse.ArgumentParser(description="Finds Pearson correlation between log returns of given assets.")

    parser.add_argument(
        "assets",
        metavar="assets",
        help="USDT denominated trading pairs.  e.g. BTCUSDT ETHUSDT",
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
        "--interval",
        metavar="",
        help="Interval for which to calculate returns as a multiple of granularity. e.g. 1 (an interval of 1 with granularity 1d would calculate returns once per day).  Default: 1",
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

    return parser.parse_args(["BTCUSDT", "ETHUSDT", "LINKUSDT", "AAVEUSDT", "MATIC", "-start", "2020-09-01", "-end", "2021-09-01"])
    # , "--granularity", "1h"



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

        dfs[asset] = pd.concat(data_lists[asset], ignore_index=True)

    return dfs




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

    return log_r_df.corr(method="pearson")




def main():
    args = get_args() 
    
    df_dict = to_dfs(args.assets, args.data_dir, args.granularity)
    # print(df_dict)
    price_df = combine_by_component(df_dict, index=None)
    corr_matrix = log_returns_corr(price_df, args.interval)
    
    print(price_df)
    # print(price_df.loc[1::1,:].reset_index(drop=True) - price_df.loc[::1,:].reset_index(drop=True))
    print(price_df - price_df.shift(1))
    print(corr_matrix)


    




if __name__ == "__main__":
    main()