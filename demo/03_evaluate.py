"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import os
import json

sys.path.append('../')

from logppt.evaluation.common import common_args
from logppt.evaluation.evaluator_main import evaluator, prepare_results
from logppt.evaluation.postprocess import post_average
import logging

logger = logging.getLogger("LogPPT")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
# set logging format
formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger.handlers[0].setFormatter(formatter)


datasets_full = [
    "Proxifier",
    "Apache",
    "OpenSSH",
    "HDFS",
    "OpenStack",
    "HPC",
    "Zookeeper",
    "HealthApp",
    "Hadoop",
    "Spark",
    "BGL",
    "Linux",
    "Mac",
    "Thunderbird",
]


if __name__ == "__main__":
    args = common_args()
    
    # initialize by data type
    datasets = datasets_full
    data_type = "full"
    input_dir = f"../datasets/loghub-{data_type}/"
    # output_dir = f"../examples/outputs/results/{args.config}" 
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

    # prepare results file
    result_file = prepare_results(
        output_dir=output_dir,
        otc=args.oracle_template_correction,
        complex=args.complex,
        frequent=args.frequent
    )
    if args.dataset != "null":
        datasets = [args.dataset]

    for dataset in datasets:
        log_file = f"{dataset}_full.log_structured.csv"
        # if os.path.exists(os.path.join(output_dir, f"{dataset}_full.log_structured.csv")):
        #     continue
        
        # run evaluator for a dataset
        # The file is only for evalutation, so we remove the parameter "LogParser"
        # train_file = f"../datasets/loghub/{dataset}/validation.json"
        train_file = f"../datasets/loghub/{dataset}/samples/logppt_32.json"
        with open(train_file, 'r') as f:
            train_data = f.readlines()
        # each line is a json object
        train_data = [json.loads(line) for line in train_data]
        filtered_logs = list(set(log['template'] for log in train_data))
        evaluator(
            dataset=dataset,
            input_dir=input_dir,
            output_dir=output_dir,
            log_file=log_file,
            otc=args.oracle_template_correction,
            complex=args.complex,
            frequent=args.frequent,
            result_file=result_file,
            filter_templates=filtered_logs,
        )  # it internally saves the results into a summary file
    metric_file = os.path.join(output_dir, result_file)
    
    if args.dataset == "null":
        post_average(metric_file)
    
