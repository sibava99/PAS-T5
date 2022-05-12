# %%
import argparse
import glob
import os
import json
import time
import logging
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

# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)



# %%
mega_tokenizer = T5Tokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
mega_model = T5ForConditionalGeneration.from_pretrained("megagonlabs/t5-base-japanese-web")
mega_model.resize_token_embeddings(len(mega_tokenizer))


# %%
dataset = load_dataset('json',data_files={"train":'/home/sibava/PAS-T5/pas-dataset-yotte/train.doc.jsonl',
"dev":'/home/sibava/PAS-T5/pas-dataset-yotte/dev.doc.jsonl'})

# %%
dataset.set_format(type='torch',columns=['input_ids','labels'])

# %%
TRAIN_BATCH_SIZE  = 16
EVAL_BATCH_SIZE  = 16
NUM_EPOCHS  = 3

training_args = TrainingArguments(
    "./finetune_yotte",
    num_train_epochs = NUM_EPOCHS,
    evaluation_strategy = "steps",
    adafactor = True,
    learning_rate=1e-3,
    lr_scheduler_type="constant",
    per_device_train_batch_size = TRAIN_BATCH_SIZE,
    per_device_eval_batch_size  = EVAL_BATCH_SIZE,
    eval_steps = 500,
    logging_steps = 500,
    save_steps = 500
)
# %%
trainer = Trainer(
    model=mega_model,
    args=training_args,
    tokenizer=mega_tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"]
)

# %%
trainer.train()
# %%
model_path = 'yotte_trained.pth'
torch.save(mega_model.state_dict(), model_path)



