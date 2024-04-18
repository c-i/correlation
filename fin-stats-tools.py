import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar import vecm
from statsmodels.tsa import stattools
from statsmodels.regression import linear_model as lin
from statsmodels.tools.tools import add_constant
from datetime import datetime as dt
from datetime import date as d
import time
import argparse
import os
from config import ABS_PATH_TO_CSV, HEADER




DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HEADER is None:
    HEADER = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]




def get_args():
    parser = argparse.ArgumentParser(description="Statistics tools for finance.")
    subparser = parser.add_subparsers(dest="stats_tool", required=True, help="stats tool")

    global_parser = subparser.add_parser("global_settings", add_help=False)
    global_parser.add_argument(
        "--start",
        help="(required) Start date in iso format.  e.g. 2020-12-30",
        required=True
    )
    global_parser.add_argument(
        "--end",
        help="(required) End date in iso format (up to the end of last month).  e.g. 2021-12-30",
        required=True
    )
    global_parser.add_argument(
        "--interval",
        help="(int) Interval for which to calculate returns as a multiple of granularity. e.g. 1 (an interval of 1 with granularity 1d would calculate returns once per day).  Default: 1",
        type=int,
        default=1
    )
    global_parser.add_argument(
        "--granularity",
        help="Granularity of k-line data.  e.g. 1d (default: 1d)",
        default="1d"
    )
    global_parser.add_argument(
        "--data_dir",
        help=f"Directory where k-line data is stored.  Default: {DIR}/spot/monthly/klines",
        default=f"{DIR}/spot/monthly/klines"
    )
    global_parser.add_argument(
        "--component",
        help="CSV header label used to retrieve data.  Default: close",
        default="close"
    )
    global_parser.add_argument(
        "--index",
        help="CSV header label used to retrieve timestamps.  Default: close_time",
        default="close_time"
    )


    correlation_parser = subparser.add_parser("correlation", help="Finds Pearson correlations between log returns of given assets.", parents=[global_parser])

    correlation_parser.add_argument(
        "assets",
        metavar="assets",
        help="USDT denominated trading pairs.  e.g. BTCUSDT ETHUSDT (if {all} is provided as the first argument all assets will be used)",
        nargs="+"
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
        "--mean_norm",
        help="(bool) Use mean normalisation on returns instead of log.",
        action="store_true",
        default=False
    )


    coint_parser = subparser.add_parser("johansen", help="Performs Johansen cointegration test on price series of given assets.", parents=[global_parser])

    coint_parser.add_argument(
        "assets",
        metavar="assets",
        help="Assets to perform the Johansen test on.",
        nargs="+"
    )
    coint_parser.add_argument(
        "-l",
        help="Number of lagged differences.  Default: 1",
        type=int,
        default=1
    )
    coint_parser.add_argument(
        "--log",
        help="Perform Johansen test on log price series.  Default: True",
        default=True
    )


    adf_parser = subparser.add_parser("adf", help="Performs augmented Dickey-Fuller test on spread between asset prices.", parents=[global_parser])

    adf_parser.add_argument(
        "assets",
        metavar="assets",
        help="Pair of assets used to calculate spread and perform the augmented Dickey-Fuller test on.",
        nargs=2
    )
    adf_parser.add_argument(
        "--beta",
        help="(required) (int) Beta term used to calculate spread.  Applied to second provided {asset} argument, ie first asset is dependent var and second is independent.  Default: 1",
        type=float,
        default=1
    )


    plot_parser = subparser.add_parser("plot", help="Plots asset data.", parents=[global_parser])

    plot_parser.add_argument(
        "assets",
        metavar="assets",
        help="Assets to plot.",
        nargs="+"
    )
    plot_parser.add_argument(
        "-r",
        help="Plot returns and mean normalised percent returns.",
        action="store_true",
        default=False
    )
    plot_parser.add_argument(
        "-sp",
        help="Plots spread between first two assets passed to {assets} argument.",
        action="store_true", 
        default=False
    )
    plot_parser.add_argument(
        "-sc",
        help="Plots scatterplot between first two assets passed to {assets} argument.",
        action="store_true",
        default=False
    )
    plot_parser.add_argument(
        "--save",
        help="Save returns and/or spread to CSV.  Save location: {output/returns} or {output/spread}. Default: False.",
        action="store_true",
        default=False
    )
    plot_parser.add_argument(
        "--beta",
        help="(int) Beta term used to calculate spread between cointegrated series.  Default: 1",
        type=float,
        default=1
    )


    ols_parser = subparser.add_parser("ols", help="Perform ordinary least squares regression on given asset price series.", parents=[global_parser])

    ols_parser.add_argument(
        "assets",
        metavar="assets",
        help="Assets to perform OLS regression on. First asset provided is the dependent variable and the second is the independent variable.",
        nargs=2
    )


    pca_parser = subparser.add_parser("pca", help="Perform principle component analysis.", parents=[global_parser])

    pca_parser.add_argument(
        "assets",
        metavar="assets",
        help="...",
        nargs="+"
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
        
    print("Reading CSVs\n\n")
    for asset in assets:
        if ABS_PATH_TO_CSV is None:
            path = f"{dir}/{asset}/{granularity}"
        else:
            path = ABS_PATH_TO_CSV

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




def percent_returns(df, interval):
    returns_df = ((df / df.shift(interval)) - 1) * 100
    return returns_df




def log_returns(df, interval):
    returns = np.log(df) - np.log(df.shift(interval))
    return returns



# per sample x: (x - mean(x)) / std(x)
def mean_normalise(df):
    normalised_df = (df - df.mean()) /df.std()
    return normalised_df




def ols(price_df):
    assets = price_df.columns

    y = price_df[assets[0]]
    x = price_df[assets[1]]
    x = add_constant(x)

    model = lin.OLS(y, x)
    results = model.fit()

    return results



# takes trading pair price dataframes and interval and returns correlation matrix of log returns (or of mean normalised returns if --mean_norm arg is passed)
def returns_corr(price_df, interval=1, mean_norm=False):
    if mean_norm:
        returns_df = percent_returns(price_df, interval)
        norm_r_df = mean_normalise(returns_df)
    else:
        norm_r_df = log_returns(price_df, interval)
    corr_matrix = norm_r_df.corr(method="pearson")

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




class Correlation:
    def __init__(self, assets, data_dir, granularity, start, end, interval, n, component, index, mean_norm):
        self.df_dict = to_dfs(assets, data_dir, granularity)
        self.price_df = combine_by_component(self.df_dict, start, end, component=component, index=index)
        self.corr_matrix = returns_corr(self.price_df, interval, mean_norm)
        self.corr_s = corr_series(self.corr_matrix)
        self.top_assets = top_assets(self.corr_s, n)
        
        sub_df_dict = to_dfs(self.top_assets, data_dir, granularity)
        sub_price_df = combine_by_component(sub_df_dict, start, end)
        self.corr_submatrix = returns_corr(sub_price_df, interval, mean_norm)

    
    def plot_heatmap(self):
        sns.heatmap(self.corr_submatrix, annot=True, cmap="viridis")
        plt.show()




def correlation(args):
    corr = Correlation(args.assets, args.data_dir, args.granularity, args.start, args.end, args.interval, args.n, args.component, args.index, args.mean_norm)

    print(corr.corr_submatrix)

    returns_type = "(percent-mean)" if args.mean_norm else "(log)"
    if args.m:
        corr.corr_matrix.to_csv(f"{DIR}/output/correlation/corr-matrix-{args.start}-to-{args.end}-{returns_type}.csv", compression=None)
        print(f"saved csv to {DIR}/output/correlation/corr-matrix-{args.start}-to-{args.end}-{returns_type}.csv")

    if args.l:
        corr.corr_s.to_csv(f"{DIR}/output/correlation/corr-list-{args.start}-to-{args.end}-{returns_type}.csv", compression=None)
        print(f"saved csv to {DIR}/output/correlation/corr-list-{args.start}-to-{args.end}-{returns_type}.csv")
    
    if args.p:
        corr.plot_heatmap()




def johansen(args):
    df_dict = to_dfs(args.assets, dir=args.data_dir, granularity=args.granularity)
    price_df = combine_by_component(df_dict, args.start, args.end, component=args.component, index=args.index)

    if args.log:
        price_df = price_df.apply(np.log)

    lag_diff = args.l
    result = vecm.coint_johansen(price_df, 0, lag_diff)

    trace_arr = np.array(result.trace_stat_crit_vals).T
    eig_arr = np.array(result.max_eig_stat_crit_vals).T
    

    results_df = pd.DataFrame(data = {"trace_stat": result.trace_stat, "trace_crit_90%": trace_arr[0], "trace_crit_95%": trace_arr[1], "trace_crit_99%": trace_arr[2], "max_eig_stat": result.max_eig_stat, "max_eig_90%": eig_arr[0], "max_eig_95%": eig_arr[1], "max_eig_99%": eig_arr[2]})

    # print("Johansen test:")
    print("lagged differences: ", lag_diff, "\n")

    # print("trace: \n", result.trace_stat, "\n\ntrace critical values: \n", result.trace_stat_crit_vals, "\n\neigenvalues: \n", result.max_eig_stat, "\n\neigenvalue crit vals: \n", result.max_eig_stat_crit_vals)
    # reject null if test > crit 
    print(results_df)




def adf(args):
    df_dict = to_dfs(args.assets, dir=args.data_dir, granularity=args.granularity)
    price_df = combine_by_component(df_dict, args.start, args.end, component=args.component, index=args.index)
    assets = price_df.columns

    spread = np.log(price_df[assets[0]]) - (args.beta * np.log(price_df[assets[1]]))

    result = stattools.adfuller(spread, maxlag=None)

    print("Results of Dickey-Fuller Test:")
    dfoutput = pd.Series(
        result[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in result[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    
    print(dfoutput)




def plot_default(price_df, returns_df, norm_r_df, r):
    for asset in list(price_df):
        sns.relplot(data=price_df, x=price_df.index, y=asset, kind="line")
        plt.title(f"{asset} price")
        plt.xticks(rotation="vertical")
        plt.xticks(price_df.index[::30])
        plt.xlabel("Date")
        

        if r:
            sns.relplot(data=returns_df, x=returns_df.index, y=asset, kind="line")
            plt.title(f"{asset} absolute percent returns")
            plt.xticks(rotation="vertical")
            plt.xticks(returns_df.index[::30])
            plt.xlabel("Date")
            
            sns.relplot(data=norm_r_df, x=norm_r_df.index, y=asset, kind="line")
            plt.title(f"{asset} mean normalised percent returns")
            plt.xticks(rotation="vertical")
            plt.xticks(norm_r_df.index[::30])
            plt.xlabel("Date")




def plot_spread(price_df, beta):
    # normalise prices
    # calculate and plot spread (use OLS?)
    assets = price_df.columns

    spread = np.log(price_df[assets[0]]) - (beta * np.log(price_df[assets[1]])) 
    spread.rename(str(beta))

    sns.relplot(data=spread, kind="line")
    plt.title(f"{assets[0]} {assets[1]} log price spread (beta= {beta})")
    plt.xticks(rotation="vertical")
    plt.xticks(price_df.index[::30])
    plt.xlabel("Date")

    return spread




def plot_scatter(price_df):
    assets = price_df.columns

    plt.figure()
    sns.scatterplot(data=price_df, x=price_df[assets[1]], y=price_df[assets[0]])
    plt.title(f"{assets[0]} {assets[1]} scatterplot")
    plt.xticks(rotation="vertical")




def plot_assets(args):
    df_dict = to_dfs(args.assets, dir=args.data_dir, granularity=args.granularity)
    price_df = combine_by_component(df_dict, args.start, args.end, component=args.component, index=args.index)

    returns_df = percent_returns(price_df, args.interval)
    norm_r_df = mean_normalise(returns_df)

    print("price:\n", price_df)
    if args.r:
        print("returns:\n", returns_df)
        print("normalised returns:\n", norm_r_df)
        print("Returns standard deviation: ", returns_df.std(axis=0))

    plot_default(price_df, returns_df, norm_r_df, args.r)

    if args.sp and price_df.shape[1] >= 2:
        spread = plot_spread(price_df, args.beta)
    if args.sp and price_df.shape[1] == 1:
        print("cannot plot spread: only 1 asset provided")
        plot_default(price_df, returns_df, norm_r_df, args.r)

    if args.sc and price_df.shape[1] >= 2:
        plot_scatter(price_df)
    if args.sc and price_df.shape[1] == 1:
        print("cannot plot spread: only 1 asset provided")
        plot_default(price_df, returns_df, norm_r_df, args.r)

    if args.save:
        if args.r:
            returns_df.to_csv(f"{DIR}/output/returns/returns-{price_df.columns}-{args.start}-to-{args.end}.csv", compression=None)
            norm_r_df.to_csv(f"{DIR}/output/returns/normalised_returns-{norm_r_df.columns}-{args.start}-to-{args.end}.csv", compression=None)

        if args.s:
            spread.to_csv(f"{DIR}/output/spread/spread-{price_df.columns}-{args.start}-to-{args.end}-(beta{spread.name}).csv", compression=None)

    plt.show()




def eigenportfolio(price_df, corr_matrix=None, interval=1):
    if corr_matrix is None:
        corr_matrix = returns_corr(price_df, interval, mean_norm=True)

    returns_df = percent_returns(price_df, interval)

    np_eigvals, np_eigvecs = np.linalg.eig(corr_matrix)
    eig_dict = dict(zip(np_eigvals.tolist(), np_eigvecs.tolist()))
    eig_dict = dict(sorted(eig_dict.items(), reverse=True))

    print(price_df)
    print(np_eigvecs)
    print(np_eigvals)

    Q_weights = []
    std_series = price_df.std()




def pca(args):
    df_dict = to_dfs(args.assets, dir=args.data_dir, granularity=args.granularity)
    price_df = combine_by_component(df_dict, args.start, args.end, component=args.component, index=args.index)

    eigenportfolio(price_df)




def main():
    if not os.path.isdir("output"):
        os.mkdir("output")

    if not os.path.isdir("output/returns"):
        os.mkdir("output/returns")

    if not os.path.isdir("output/spread"):
        os.mkdir("output/spread")

    if not os.path.isdir("output/correlation"):
        os.mkdir("output/correlation")


    args = get_args() 

    if args.stats_tool == "correlation":
        correlation(args)

    if args.stats_tool == "johansen":
        johansen(args)

    if args.stats_tool == "adf":
        adf(args)

    if args.stats_tool == "plot":
        plot_assets(args)

    if args.stats_tool == "ols":
        df_dict = to_dfs(args.assets, dir=args.data_dir, granularity=args.granularity)
        price_df = combine_by_component(df_dict, args.start, args.end, component=args.component, index=args.index)

        result = ols(price_df)
        print(result.summary())

    if args.stats_tool == "pca":
        pca(args)


if __name__ == "__main__":
    main()




# TODO: implement async file io (way too much effort, pandas read_csv is not asynchronous, would have to unload csvs with csv lib ? or some async file io lib)
