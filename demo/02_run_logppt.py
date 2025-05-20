"""
This script demonstrates how to run the LogPPT model for log parsing.
```bash
cd demo
export dataset=Apache
python 02_run_logppt.py --log_file ../datasets/loghub-full/$dataset/${dataset}_full.log_structured.csv --model_name_or_path roberta-base --train_file ../datasets/loghub-full/$dataset/samples/logppt_32.json --validation_file ../datasets/loghub-full/$dataset/validation.json --dataset_name $dataset --parsing_num_processes 4 --output_dir ./results/models/$dataset --task_output_dir ./results/logs --max_train_steps 1000
```
"""

import sys
sys.path.append("../")

from collections import Counter
import time
from transformers import set_seed
import logging
import os
import pandas as pd
import torch
import json

from logppt.data.data_loader import DataLoaderForPromptTuning
from logppt.models.roberta import RobertaForLogParsing
from logppt.trainer import TrainingArguments, Trainer
from logppt.parsing_base import template_extraction
from logppt.arguments import get_args


device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger = logging.getLogger("LogPPT")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
# set logging format
formatter = logging.Formatter(
    '%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger.handlers[0].setFormatter(formatter)


if __name__ == '__main__':

    data_args, model_args, train_args, common_args = get_args()

    if common_args.seed is not None:
        set_seed(common_args.seed)

    assert data_args.train_file is not None, "A training file is required"

    # Load data
    logger.info("Loading training data from {0} ...".format(
        data_args.train_file))
    data_loader = DataLoaderForPromptTuning(data_args)

    # Load model and tokenizer
    p_model = RobertaForLogParsing(model_args.model_name_or_path, use_crf=model_args.use_crf)
    logger.info("Loaded RoBERTa model")

    p_model.tokenizer = data_loader.initialize(p_model.tokenizer)
    data_loader.tokenize()
    data_loader.build_dataloaders(
        train_args.per_device_train_batch_size, train_args.per_device_eval_batch_size)
    param_label_words = data_loader.find_parameter_label_words(
        p_model.plm, common_args.no_label_words)
    p_model.add_label_token(param_label_words)

    # Training
    training_args = TrainingArguments(
        output_dir=common_args.output_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        do_predict=False,
        evaluation_strategy="steps",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        num_train_epochs=train_args.num_train_epochs,
        max_train_steps=train_args.max_train_steps,
        gradient_accumulation_steps=1,
        lr_scheduler_type="polynomial"
    )

    trainer = Trainer(
        model=p_model,
        args=training_args,
        train_loader=data_loader.get_train_dataloader(),
        eval_loader=data_loader.get_val_dataloader(),
        compute_metrics=None,
        no_train_samples=len(data_loader.raw_datasets['train']),
        device=device
    )
    
    # Training
    t0 = time.time()
    p_model = trainer.train()
    trainer.save_pretrained(common_args.output_dir)
    t1 = time.time()
    logger.info("Training time: {0} seconds".format(t1 - t0))

    # Parsing
    p_model.load_checkpoint(common_args.output_dir)
    start_time = time.time()
    logger.info("Loading log file {0} ...".format(data_args.log_file))
    log_df = pd.read_csv(data_args.log_file)
    log_lines = log_df['Content'].tolist()

    p_model.eval()

    # cache_lock = multiprocessing.Lock()
    # templates, model_time, template_list = template_extraction(
    #     p_model, device, log_lines, vtoken=data_loader.vtoken, n_workers=common_args.parsing_num_processes)
    templates, model_time = template_extraction(
        p_model, device, log_lines, vtoken=data_loader.vtoken)

    # templates = [template[0] if template[1]
    #              is None else template_list[template[1]] for template in templates]

    log_df['EventTemplate'] = pd.Series(templates)

    # Save the results
    t2 = time.time()
    task_output_dir = data_args.task_output_dir
    if not os.path.exists(task_output_dir):
        os.makedirs(task_output_dir)
    
    # save common_args, training_args, model_args, data_args to config.json
    if not os.path.exists(f"{task_output_dir}/config.json"):
        with open(f"{task_output_dir}/config.json", 'w') as file:
            json.dump({
                'common_args': vars(common_args),
                'training_args': vars(training_args),
                'model_args': vars(model_args),
                'data_args': vars(data_args)
            }, file)
    log_df.to_csv(
        f"{task_output_dir}/{data_args.dataset_name}_full.log_structured.csv", index=False)

    counter = Counter(templates)
    items = list(counter.items())
    items.sort(key=lambda x: x[1], reverse=True)
    template_df = pd.DataFrame(items, columns=['EventTemplate', 'Occurrence'])
    template_df['EventID'] = [f"E{i + 1}" for i in range(len(template_df))]
    template_df[['EventID', 'EventTemplate', 'Occurrence']].to_csv(
        f"{task_output_dir}/{data_args.dataset_name}_full.log_templates.csv", index=False)

    # Save time cost
    time_cost_file = f"{task_output_dir}/time_cost.json"
    time_table = {}
    if os.path.exists(time_cost_file):
        with open(time_cost_file, 'r') as file:
            time_table = json.load(file)
    time_table[data_args.dataset_name] = {
        'TrainingTime': (t1 - t0).__round__(3),
        'ParsingTime': (t2 - t1).__round__(3),
        'ModelTime': model_time.__round__(3)
    }
    with open(time_cost_file, 'w') as file:
        json.dump(time_table, file)
