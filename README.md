CLI tool that calculates the Pearson correlation between the log returns of cryptocurrencies and outputs a correlation matrix and list of correlations sorted from strongest to weakest, with options to save the correlation matrix and/or correlation list to a csv file, and an option to plot a correlation matrix heatmap.

Requirements: numpy, pandas, matplotlib, seaborn, python >=3.10

Price data: This tool is intended to be used with spot k-line data downloaded using the stas-prokopiev/binance_historical_data python module.  This module saves spot data to the directory /spot/.  /spot/ should be placed in the same directory as correlation.py.  My binance-data-get CLI wrapper for the binance-historical-data module makes it easy to download all Binance k-line data at once.  You may want to remove all non-token contract data such as UP and DOWN contracts before looking for correlations.  Modifying the tool to be compatible with other exchanges or non-k-line data is quite trivial: just modify the directory of where the data CSVs are located and modify the HEADERS constant to match the CSV headers of the new data.

Output: CSV files are stored in the /output/ directory. 


    usage: correlation.py [-h] --start START --end END [-p] [-n N] [-m] [-l] [--interval INTERVAL]
                          [--granularity GRANULARITY] [--data_dir DATA_DIR] [--component COMPONENT] [--index INDEX]
                          assets [assets ...]
    
    Finds Pearson correlation between log returns of given assets.
    
    positional arguments:
      assets                USDT denominated trading pairs. e.g. BTCUSDT ETHUSDT (if {all} is provided as the first
                            argument all assets will be used)
    
    options:
      -h, --help            show this help message and exit
      --start START         (required) Start date in iso format. e.g. 2020-12-30
      --end END             (required) End date in iso format (up to the end of last month). e.g. 2021-12-30
      -p                    (bool) Choose whether to plot correlation matrix heatmap of top n assets.
      -n N                  (int) Number of assets to plot in correlation matrix heatmap. Default: 10
      -m                    (bool) Save correlation matrix to csv file.
      -l                    (bool) Save list of asset pair correlations sorted from greatest to least to csv file.
      --interval INTERVAL   (int) Interval for which to calculate returns as a multiple of granularity. e.g. 1 (an
                            interval of 1 with granularity 1d would calculate returns once per day). Default: 1
      --granularity GRANULARITY
                            Granularity of k-line data. e.g. 1d (default: 1d)
      --data_dir DATA_DIR   Directory where k-line data is stored. Default:
                            C:/Users/chuck/Documents/projects/statarb/correlation/spot/monthly/klines
      --component COMPONENT
                            CSV header label used to calculate log returns. Default: close
      --index INDEX         CSV header label used to retrieve timestamps. Default: close_time
