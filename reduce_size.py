import concurrent.futures
import datetime
import pandas as pd
from glob import glob
from tardis_dev import datasets, get_exchange_details
import os

COLUMNS = ["symbol", "timestamp", "type", "strike_price", "expiration", "underlying_price", "bid_price", "bid_amount",
           "ask_price", "ask_amount", "mark_price", "mark_iv", "delta", "theta"]

COLUM_TYPES = {
    'symbol': str,
    'timestamp': int,
    'type': str,
    'strike_price': float,
    'expiration': int,
    'underlying_price': float,
    'bid_price': float,
    'bid_amount': float,
    'ask_price': float,
    'ask_amount': float,
    'mark_price': float,
    'delta': float,
    'theta': float
}

def get_files(folder='datasets'):
    """Get the *.csv.gz files from the folder"""
    files = glob(f"{folder}/*.csv.gz")
    files.sort()
    return files

# no real reason to pick this threshold.
def filter_expiration(df, min=0, max=90):
    """Returns the subset of options where the time to expiration is between min and max"""
    return df[(df['days_to_expiration'] > datetime.timedelta(days=min)) & (
                df['days_to_expiration'] < datetime.timedelta(days=max))]


def _load_file_to_dataframe(file_path):
    df = pd.read_csv(file_path, compression='gzip', usecols=COLUMNS,
                    dtype=COLUM_TYPES, engine='c', float_precision="legacy")
                    
    eth_only = df[df['symbol'].str.contains("ETH")]
    eth_only['datetime'] = pd.to_datetime(eth_only['timestamp'] * 1000)
    eth_only['expiration_datetime'] = pd.to_datetime(eth_only['expiration'] * 1000)
    eth_only['days_to_expiration'] = eth_only['expiration_datetime'] - eth_only['datetime']
    eth_only.dropna(inplace=True)

    return filter_expiration(eth_only)


def reduce_data(a_file:str):
    """Thead safe read and reduce the size of the options data"""
    save_path = f'reduced_datasets/reduced_{a_file.split("/")[1]}'
    df = _load_file_to_dataframe(a_file)
    df.to_csv(save_path, compression='gzip')
    return (save_path, df.shape)

def get_data():
    """Save the options data from deribit. Not you need to change the API key here"""
    API_KEY='my api key'
    get_exchange_details("deribit")
    datasets.download(
        exchange="deribit",
        data_types=["options_chain"],
        from_date="2021-09-24", 
        to_date="2022-04-20", 
        symbols=["OPTIONS"],
        api_key=API_KEY,
    ) 

def select_un_reduced_files():
    """Returns the files in datasets that have not been reduced yet"""
    files_in_reduced_data = get_files('reduced_datasets')
    files_in_datasets = get_files('datasets')
    files_to_save = []
    for a_file in files_in_datasets:
        save_path = f'reduced_datasets/reduced_{a_file.split("/")[1]}'
        if save_path not in files_in_reduced_data:
            files_to_save.append(a_file)
    return files_to_save

def concurrent_reduce_size(files_to_save:list[str], n_workers=8) -> None:
    """
    Reduce the size of files_to_save with n_worker threads. 
    n_workers depends on how much ram you have. 8 workers uses about 45 gb of ram.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor: 
        futures = []
        for a_file in files_to_save:
            futures.append(executor.submit(reduce_data, a_file=a_file))
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

def main():
    get_data() # you need to put in the api key to get this to run
    try:
        os.mkdir('reduced_datasets')
    except FileExistsError:
        pass
    
    files_to_save = select_un_reduced_files()
    print(f'You have {len(files_to_save)} files to save')
    concurrent_reduce_size(files_to_save, 8)


if __name__ == '__main__':
    main() 