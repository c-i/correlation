import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



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


    def plot_heatmap(corr_matrix):
        sns.heatmap(corr_matrix, annot=True, cmap="viridis")
        plt.show()

