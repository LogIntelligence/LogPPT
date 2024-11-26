import sys
sys.path.append('.')

import json
import os
import pandas as pd
from logppt.sampling_base import adaptive_random_sampling
from logppt import BENCHMARK


def test_adaptive_random_sampling(data_version='2k'):
    data_dir = f"./datasets/loghub-{data_version}"
    output_dir = f"./datasets/loghub-{data_version}"
    for dataset in BENCHMARK.keys():
        print(dataset)
        if dataset != 'Apache':
            continue
        os.makedirs(f'{output_dir}/{dataset}/samples', exist_ok=True)
        log_file = BENCHMARK[dataset]['log_file'].format(data_version)
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

        ## Adaptive Random Sampling from LogPPT ###
        sample_candidates = adaptive_random_sampling(raw_logs, labels, shots)

        for shot, samples in sample_candidates.items():
            assert len(samples) == shot, f"Sample size mismatch: {len(samples)} != {shot}"
            with open(f'{output_dir}/{dataset}/samples/logppt_{shot}.json', 'w') as f:
                for sample in samples:
                    f.write(json.dumps({'log': sample[0], 'template': sample[1]}) + '\n')

if __name__ == '__main__':
    test_adaptive_random_sampling(data_version='full')