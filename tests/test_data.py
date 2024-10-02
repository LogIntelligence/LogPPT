import sys
sys.path.append('.')

from logppt.data import load_loghub_dataset, DATASET_FORMAT

def test_load_csv(dataset='Apache'):
    print(f"Loading {dataset}...")
    log_df = load_loghub_dataset(dataset, cache_dir="./datasets/loghub-2.0", format='csv')
    assert log_df is not None, f"Failed to load {dataset}."

def test_load_text(dataset='Apache'):
    print(f"Loading {dataset}...")
    log_df = load_loghub_dataset(dataset, cache_dir="./datasets/loghub-2.0", format='text', log_format=DATASET_FORMAT[dataset])
    assert log_df is not None, f"Failed to load {dataset}."