import os
from operator import itemgetter
import pickle

CORRUPT_FILE = 'corrupted.pkl'
RESULTS_FILE = 'results.pkl'
def add_to_corrupted_files(file_path):
    corrupted = set()
    if os.path.exists(CORRUPT_FILE) and os.path.getsize(CORRUPT_FILE) > 0:
        corrupted = get_corrupted_files()
    corrupted.add(file_path)
    with open(CORRUPT_FILE,'wb') as wfp:
        pickle.dump(corrupted, wfp)

def add_to_results_files(stats):
    results = []
    if os.path.exists(RESULTS_FILE) and os.path.getsize(RESULTS_FILE) > 0:
        results = get_data_from_file(RESULTS_FILE)
    results.append(stats)
    with open(RESULTS_FILE,'wb') as wfp:
        pickle.dump(results, wfp)

def remove_corrupted_file(file):
    files = get_corrupted_files()
    files.remove(file)
    with open(CORRUPT_FILE,'wb') as wfp:
        pickle.dump(files, wfp)

def get_corrupted_files():
    files = set()
    with open(CORRUPT_FILE, 'rb') as f:
        files = pickle.load(f)
    return files

def get_data_from_file(file):
    data = None
    with open(file, 'rb') as f:
        files = pickle.load(f)
    return data

def get_date_from_filename(fn):
    return fn.split('/')[-1].split('_')[-2]

def get_sorted_corrupt_list():
    files_set = get_corrupted_files()
    files = list(files_set)
    col = [{'filename': x, 'date': get_date_from_filename(x) } for x in files]
    return sorted(col, key=itemgetter('date'))
