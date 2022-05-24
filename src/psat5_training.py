
import argparse
import glob
import os
import json
import time
import logzero
from logzero import logger
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Config
)
from transformers.optimization import Adafactor, AdafactorSchedule

# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def create_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=str, required=True,
                        help="Path to train dataset'")
    parser.add_argument('--dev', type=str, required=True,
                        help="Path to dev dataset'")
    parser.add_argument('--output',type=str, required=True,
                        help="Path to output directory.")
    parser.add_argument('--lr',type=float, required=True,
                        help="learning rate")
    parser.add_argument('--batch_size',type=int, required=True,
                        help="number of batch size")
    parser.add_argument('--epoch',type=int, required=True,
                        help="number of epoch")

    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    output = args.output
    os.mkdir(output)
    logzero.logfile(os.path.join(output,'trainig.log'))
    logger.info(args)
    mega_tokenizer = T5Tokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
    mega_model = T5ForConditionalGeneration.from_pretrained("megagonlabs/t5-base-japanese-web")
    mega_model.resize_token_embeddings(len(mega_tokenizer))

    dataset = load_dataset('json',data_files={"train":args.train,
    "dev":args.dev})

    dataset.set_format(type='torch',columns=['input_ids','labels'])

    TRAIN_BATCH_SIZE  = args.batch_size
    EVAL_BATCH_SIZE  = args.batch_size
    NUM_EPOCHS  = args.epoch

    training_args = TrainingArguments(
        output,
        num_train_epochs = NUM_EPOCHS,
        evaluation_strategy = "steps",
        adafactor = True,
        learning_rate=1e-4,
        lr_scheduler_type="constant",
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        per_device_eval_batch_size  = EVAL_BATCH_SIZE,
        eval_steps = 500,
        logging_steps = 500,
        save_steps = 500
    )
    trainer = Trainer(
        model=mega_model,
        args=training_args,
        tokenizer=mega_tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"]
    )

    trainer.train()

    model_path = os.path.join(output,'final_model.pth')
    torch.save(mega_model.state_dict(), model_path)

if __name__ == "__main__":
    main()
    logger.info("done")

