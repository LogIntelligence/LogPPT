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

import json
import os
import re
import time
import csv
from multiprocessing import Process

from tqdm import tqdm
from logppt.evaluation.common import correct_templates_and_update_files
from logppt.evaluation.GA_calculator import evaluate
from logppt.evaluation.template_level_analysis import evaluate_template_level, evaluate_template_level_lstm
from logppt.evaluation.PA_calculator import calculate_parsing_accuracy, calculate_parsing_accuracy_lstm
import pandas as pd
from .post_process import correct_single_template

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)

import logging

logger = logging.getLogger("LogPPT")

# TIMEOUT = 3600 * 12  # log template identification timeout (sec)
TIMEOUT = 3600 * 48  # log template identification timeout (sec)


def prepare_results(output_dir, otc, complex, frequent):
    if not os.path.exists(output_dir):
        # make output directory
        os.makedirs(output_dir)

    # make a new summary file
    result_file = 'summary_[otc={},complex={},frequent={}].csv'.format(str(otc), str(int(complex)), str(int(frequent)))
    if not os.path.exists(os.path.join(output_dir, result_file)):
        with open(os.path.join(output_dir, result_file), 'w') as csv_file:
            fw = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            # fw.writerow(['Dataset', 'GA_time', 'PA_time', 'TA_time', 'parse_time', 'identified_templates',
            #              'ground_templates', 'GA', 'PA', 'FTA', 'PTA', 'RTA', 'OG', 'UG', 'MX'])
            fw.writerow(['Dataset', 'traning_time', 'parsing_time', 'identified_templates',
                        'ground_templates', 'GA', 'PA', 'FGA', 'FTA', 'uPA', 'PTA', 'RTA'])

    return result_file


def correct_template_general(template):
    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    #logger.info("CV: ", template)
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        template = re.sub(r'<\*>\:<\*>', '<*>', template)
        template = re.sub(r'<\*> <\*>', '<*>', template)
        if prev == template:
            break
    # while "<*>:<*>" in template:
    #     template = template.replace("<*>:<*>", "<*>")

    return template

def align_with_null_values(groudtruth_row):
    """
    Align the null values in the groundtruth with Content.
    """

    log = groudtruth_row['Content']
    template = groudtruth_row['EventTemplate']

    pattern_parts = template.split("<*>")
    pattern_parts_escaped = [re.escape(part) for part in pattern_parts]
    regex_pattern = "(.*?)".join(pattern_parts_escaped)
    regex = "^" + regex_pattern + "$"  
    matches = re.search(regex, log)

    if matches == None:
        return template

    parts = []
    for index, part in enumerate(template.split("<*>")):
        parts.append(part)
        if index < len(matches.groups()):
            if matches.groups()[index] == '':
                parts.append('')
            else:
                parts.append('<*>')
    return ''.join(parts)

def is_file_empty(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        return len(content) == 0


def evaluator(
        dataset,
        input_dir,
        output_dir,
        log_file,
        otc,
        complex,
        frequent,
        result_file,
        lstm=False,
        filter_templates=None
):
    """
    Unit function to run the evaluation for a specific configuration.

    """

    logger.info('\n=== Evaluation on %s ===' % dataset)
    indir = os.path.join(input_dir, os.path.dirname(log_file))
    log_file_basename = os.path.basename(log_file)
    logger.info(log_file_basename)
    groundtruth = os.path.join(indir, f"{dataset}/{log_file_basename}")
    parsedresult = os.path.join(output_dir, log_file_basename)

    # if not os.path.exists(parsedresult):
    #     with open(parsedresult, 'w') as fw:
    #         pass
    logger.info(parsedresult)
    if not os.path.exists(parsedresult) or is_file_empty(parsedresult):
        logger.info("No output file generated.")
        result = dataset + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + ',' + \
                "None" + '\n'
                # "{:.1f}".format(GA_end_time) + ',' + \
                # "{:.1f}".format(PA_end_time) + ',' + \
                # "{:.1f}".format(TA_end_time) + ',' + \

        with open(os.path.join(output_dir, result_file), 'a') as summary_file:
            summary_file.write(result)
        return   

    parsedresult = pd.read_csv(parsedresult, dtype=str)
    parsedresult.fillna("", inplace=True)
    groundtruth = pd.read_csv(groundtruth, dtype=str)
    parsedresult.Content = parsedresult.Content.str.lower()
    groundtruth.Content = groundtruth.Content.str.lower()
    parsedresult.EventTemplate = parsedresult.EventTemplate.str.lower()
    groundtruth.EventTemplate = groundtruth.EventTemplate.str.lower()
    logger.info("Start to modify output")
    parsedresult['EventTemplate'] = parsedresult['EventTemplate'].parallel_apply(lambda x: correct_single_template(x))
    groundtruth['EventTemplate'] = groundtruth['EventTemplate'].parallel_apply(lambda x: correct_single_template(x))

    # remove null values
    tqdm.pandas()
    logger.info("Start to align with null values")
    groundtruth['EventTemplate'] = groundtruth.parallel_apply(align_with_null_values, axis=1)
    # groundtruth['EventTemplate'] = groundtruth['EventTemplate'].map(correct_template_general)
    parsedresult['EventTemplate'] = parsedresult.parallel_apply(align_with_null_values, axis=1)
    if filter_templates is not None:
        # apply correct_template_general to filter_templates
        filter_templates = [correct_single_template(template.lower()) for template in filter_templates]


    logger.info("Start compute grouping accuracy")
    # calculate grouping accuracy
    start_time = time.time()
    GA, FGA = evaluate(groundtruth, parsedresult)

    GA_end_time = time.time() - start_time
    logger.info('Grouping Accuracy calculation done. [Time taken: {:.3f}]'.format(GA_end_time))

    # calculate parsing accuracy
    start_time = time.time()
    if lstm == True:
        PA = calculate_parsing_accuracy_lstm(groundtruth, parsedresult, filter_templates)
        logger.info("Finish calculate_parsing_accuracy_lstm")
    else:
        PA, uPA = calculate_parsing_accuracy(groundtruth, parsedresult, filter_templates)
    PA_end_time = time.time() - start_time
    logger.info('Parsing Accuracy calculation done. [Time taken: {:.3f}]'.format(PA_end_time))

    # calculate template-level accuracy
    start_time = time.time()
    if lstm == True:
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level_lstm(dataset, groundtruth, parsedresult)
    else:
        tool_templates, ground_templates, FTA, PTA, RTA = evaluate_template_level(dataset, groundtruth, parsedresult)
    TA_end_time = time.time() - start_time
    logger.info('Template-level accuracy calculation done. [Time taken: {:.3f}]'.format(TA_end_time))

    # read the time cost
    time_cost_file = os.path.join(output_dir, 'time_cost.json')
    parsing_time, training_time = 0, 0
    if os.path.exists(time_cost_file):
        with open(time_cost_file, 'r') as file:
            time_table = json.load(file)
            training_time = time_table[dataset]['TrainingTime']
            parsing_time = time_table[dataset]['ParsingTime']

    result = dataset + ',' + \
             "{:.3f}".format(training_time) + ',' + \
             "{:.3f}".format(parsing_time) + ',' + \
             str(tool_templates) + ',' + \
             str(ground_templates) + ',' + \
             "{:.3f}".format(GA) + ',' + \
             "{:.3f}".format(PA) + ',' + \
             "{:.3f}".format(FGA) + ',' + \
             "{:.3f}".format(FTA) + ',' + \
             "{:.3f}".format(uPA) + ',' + \
             "{:.3f}".format(PTA) + ',' + \
             "{:.3f}".format(RTA) + '\n'
             # "{:.1f}".format(GA_end_time) + ',' + \
             # "{:.1f}".format(PA_end_time) + ',' + \
             # "{:.1f}".format(TA_end_time) + ',' + \

    with open(os.path.join(output_dir, result_file), 'a') as summary_file:
        summary_file.write(result)
