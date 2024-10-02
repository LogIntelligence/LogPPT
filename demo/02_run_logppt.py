"""
This script demonstrates how to run the LogPPT model for log parsing.
```bash
cd demo
export dataset=Apache
python 02_run_logppt.py --log_file ../datasets/loghub-2k/$dataset/${dataset}_full.log_structured.csv --model_name_or_path roberta-base --train_file ../datasets/loghub-2k/$dataset/samples/logppt_32.json --validation_file ../datasets/loghub-2k/$dataset/validation.json --dataset_name $dataset --parsing_num_processes 1 --output_dir ./results/models/$dataset --task_output_dir ./results/logs/$dataset --max_train_steps 100
```
"""

import torch
import pandas as pd
from logppt.arguments import get_args
import logging
from transformers import set_seed
from logppt.parsing_par import template_extraction_concurrent
from logppt.trainer import TrainingArguments, Trainer
from logppt.models.roberta import RobertaForLogParsing
from logppt.data.data_loader import DataLoaderForPromptTuning
import os
import time
import sys
sys.path.append("../")


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

    # Load model and tokenizer
    p_model = RobertaForLogParsing(
        model_args.model_name_or_path, ct_loss_weight=0.1)

    logger.info("Loaded RoBERTa model")

    # Load data
    logger.info("Loading training data from {0} ...".format(
        data_args.train_file))
    data_loader = DataLoaderForPromptTuning(data_args)
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
        do_eval=True,
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
        device=device,
        # logger=logger
    )

    t0 = time.time()
    p_model = trainer.train()
    trainer.save_pretrained(common_args.output_dir)
    t1 = time.time()
    logger.info("Training time: {0} seconds".format(t1 - t0))

    # Parsing
    start_time = time.time()
    logger.info("Loading log file {0} ...".format(data_args.log_file))
    log_lines = pd.read_csv(data_args.log_file)['Content'].tolist()

    p_model.eval()
    templates, model_time = template_extraction_concurrent(
        p_model, device, log_lines, vtoken=data_loader.vtoken, n_workers=common_args.parsing_num_processes)
