#!/bin/bash

# This script is used to benchmark the performance of LogPPT+

for dataset in "Apache" "BGL" "HPC" "Hadoop" "HealthApp" "Linux" "Mac" "OpenStack" "Proxifier" "Spark" "Thunderbird" "Windows" "Zookeeper" "Thunderbird"; do
    echo "Processing $dataset..."
    python 02_run_logppt.py --log_file ../datasets/loghub-full/$dataset/${dataset}_full.log_structured.csv --model_name_or_path roberta-base --train_file ../datasets/loghub-full/$dataset/samples/logppt_32.json --validation_file ../datasets/loghub-full/$dataset/validation.json --dataset_name $dataset --parsing_num_processes 4 --output_dir ./results/models/$dataset --task_output_dir ./results/logs --max_train_steps 1000 --use_crf
done
