CLI tool that calculates the Pearson correlation between the log returns of cryptocurrencies and outputs a correlation matrix and list of correlations sorted from strongest to weakest, with options to save the correlation matrix and/or correlation list to a csv file, and an option to plot a correlation matrix heatmap.

Requirements: numpy, pandas, matplotlib, seaborn, python >=3.10

Price data: This tool is intended to be used with spot k-line data downloaded using the stas-prokopiev/binance_historical_data python module.  This module saves spot data to the directory /spot/.  /spot/ should be placed in the same directory as correlation.py.  My binance-data-get CLI wrapper for the binance-historical-data module makes it easy to download all Binance k-line data at once.  You may want to remove all non-token contract data such as UP and DOWN contracts before looking for correlations.  Modifying the tool to be compatible with other exchanges or non-k-line data is quite trivial: just modify the directory of where the data CSVs are located and modify the HEADERS constant to match the CSV headers of the new data.

Output: CSV files are stored in the /output/ directory. 

Usage: python correlaton.py -h
