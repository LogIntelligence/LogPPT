import sys
sys.path.append("../")

import json
import os
import pandas as pd

from logppt import BENCHMARK
from logppt.sampling_base import adaptive_random_sampling
from logppt.sampling.hierachical_sampling import sampling

DATA_VERSION = 'full' # '2k' or 'full'

if __name__ == '__main__':
    data_dir = f"../datasets/loghub-{DATA_VERSION}"
    output_dir = f"../datasets/loghub-{DATA_VERSION}"
    for dataset in BENCHMARK.keys():
        print(dataset)
        os.makedirs(f'{output_dir}/{dataset}/samples', exist_ok=True)
        log_file = BENCHMARK[dataset]['log_file'].format(DATA_VERSION)
        if not os.path.exists(f'{data_dir}/{log_file}_structured.csv'):
            print(f"File {log_file}_structured.csv does not exist. Skipping...")
            continue
        print(f"Loading {log_file}...")
        labelled_logs = pd.read_csv(f'{data_dir}/{log_file}_structured.csv')
        print(f"Loaded {len(labelled_logs)} logs.")
        k_rate = 0.2
        length = int(k_rate * len(labelled_logs))
        labelled_logs = labelled_logs[:length]
        raw_logs = labelled_logs['Content'].tolist()
        labels = labelled_logs['EventTemplate'].tolist()
        with open(f'{output_dir}/{dataset}/validation.json', 'w') as f:
            for log, label in zip(raw_logs, labels):
                f.write(json.dumps({'log': log, 'template': label}) + '\n')
        shots = [8, 16, 32, 64, 128, 256]

        ### Hierarchical sampling
        sample_candidates = sampling(raw_logs, labels, shots)

        for shot, samples in sample_candidates.items():
            assert len(samples) == shot, f"Sample size mismatch: {len(samples)} != {shot}"
            with open(f'{output_dir}/{dataset}/samples/logppt_{shot}.json', 'w') as f:
                for sample in samples:
                    f.write(json.dumps({'log': sample[0], 'template': sample[1]}) + '\n')

        ### Adaptive random sampling
        sample_candidates = adaptive_random_sampling(raw_logs, labels, shots, n_candidate=32)
        for shot, samples in sample_candidates.items():
            assert len(samples) == shot, f"Sample size mismatch: {len(samples)} != {shot}"
            with open(f'{output_dir}/{dataset}/samples/random_{shot}.json', 'w') as f:
                for sample in samples:
                    f.write(json.dumps({'log': sample[0], 'template': sample[1]}) + '\n')
