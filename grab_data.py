# pip install tardis-dev
# requires Python >=3.6
from tardis_dev import datasets, get_exchange_details
import logging
import os
from dotenv import load_dotenv
import argparse, sys
from storage import get_sorted_corrupt_list, remove_corrupted_file
import wdb

load_dotenv()
api_key = os.getenv('API_KEY')

parser=argparse.ArgumentParser()
parser.add_argument('--fix', help='Download only files determined corrupted', action='store_true')
parser.set_defaults(fix=False)
args=parser.parse_args()

# comment out to disable debug logs
logging.basicConfig(level=logging.DEBUG)

# function used by default if not provided via options
def default_file_name(exchange, data_type, date, symbol, format):
    return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# customized get filename function - saves data in nested directory structure
def file_name_nested(exchange, data_type, date, symbol, format):
    return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# returns data available at https://api.tardis.dev/v1/exchanges/deribit
deribit_details = get_exchange_details("deribit")
# print(deribit_details)

def download():
    datasets.download(
    # one of https://api.tardis.dev/v1/exchanges with supportsDatasets:true - use 'id' value
    exchange="deribit",
    # accepted data types - 'datasets.symbols[].dataTypes' field in https://api.tardis.dev/v1/exchanges/deribit,
    # or get those values from 'deribit_details["datasets"]["symbols][]["dataTypes"] dict above
    data_types=["options_chain"],
    # change date ranges as needed to fetch full month or year for example
    from_date="2021-05-26",
    # to date is non inclusive
    to_date="2021-09-25",
    # accepted values: 'datasets.symbols[].id' field in https://api.tardis.dev/v1/exchanges/deribit
    symbols=["OPTIONS"],
    # (optional) your API key to get access to non sample data as well
    api_key=api_key,
    # (optional) path where data will be downloaded into, default dir is './datasets'
    #download_dir="./option_chain",
    # (optional) - one can customize downloaded file name/path (flat dir strucure, or nested etc) - by default function 'default_file_name' is used
    # get_filename=default_file_name,
    # (optional) file_name_nested will download data to nested directory structure (split by exchange and data type)
    # get_filename=file_name_nested,
    )

def download_file(date):
    datasets.download(
    # one of https://api.tardis.dev/v1/exchanges with supportsDatasets:true - use 'id' value
    exchange="deribit",
    # accepted data types - 'datasets.symbols[].dataTypes' field in https://api.tardis.dev/v1/exchanges/deribit,
    # or get those values from 'deribit_details["datasets"]["symbols][]["dataTypes"] dict above
    data_types=["options_chain"],
    # change date ranges as needed to fetch full month or year for example
    from_date=date,
    # to date is non inclusive
    to_date=date,
    # accepted values: 'datasets.symbols[].id' field in https://api.tardis.dev/v1/exchanges/deribit
    symbols=["OPTIONS"],
    # (optional) your API key to get access to non sample data as well
    api_key=api_key,
    # (optional) path where data will be downloaded into, default dir is './datasets'
    #download_dir="./option_chain",
    # (optional) - one can customize downloaded file name/path (flat dir strucure, or nested etc) - by default function 'default_file_name' is used
    # get_filename=default_file_name,
    # (optional) file_name_nested will download data to nested directory structure (split by exchange and data type)
    # get_filename=file_name_nested,
    )

def fix_corrupted_files():
    files = get_sorted_corrupt_list()
    for file in files:
        filename = file['filename']
        print(f'starting fix of {filename}')
        print(file)
        try:
            os.remove(file['filename'][1:])
        except:
            print(f'file not found')
        download_file(file['date'])
        remove_corrupted_file(file['filename'])
        print(f'file {filename} re-downloaded')

if args.fix:
    fix_corrupted_files()
else:
    download
