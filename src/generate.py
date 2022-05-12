# %%
import json
import time
import pprint
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5Config
)
import MeCab
import unidic
import re

# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = 'cuda:0'

# %%
mega_tokenizer = T5Tokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
mega_model = T5ForConditionalGeneration.from_pretrained("megagonlabs/t5-base-japanese-web").to(device)
mega_model.resize_token_embeddings(len(mega_tokenizer))
mega_model.load_state_dict(torch.load('closest_trained.pth'))

# %%
mega_model.eval()

# %%
case_en_ja = {
	'ga':'が',
	'o':'を',
	'ni':'に',
	'yotte':'によって'
}

# %%
from pyknp import Jumanpp
jumanpp = Jumanpp()

# %%
def replace_t5tok_2_sptok(t5_text:str) -> str:
	t5_text = t5_text.replace('<pad>','')
	t5_text = t5_text.replace('</s>','')
	t5_text = t5_text.replace('<extra_id_99>','→')
	t5_text = t5_text.replace('<extra_id_98>','↑')
	t5_text = t5_text.replace('<extra_id_97>','←')
	return t5_text
def replace_sptok_2_t5tok(sp_text:str) -> str:
	sp_text = sp_text.replace('→','<extra_id_99>')
	sp_text = sp_text.replace('↑','<extra_id_98>')
	sp_text = sp_text.replace('←','<extra_id_97>')
	return sp_text

# %%
output = mega_model.generate(mega_tokenizer('私は彼にボールを<extra_id_0>投げた<extra_id_1>',return_tensors="pt").input_ids.to(device))

# %%
def extrace_argument(output_ids:list,case:str,is_alt:bool=False) -> str:
    sep_id = mega_tokenizer.encode('<extra_id_2>',add_special_tokens=None)[0]
    if(is_alt):
        if(sep_id in output_ids.tolist()):
            sep_index = output_ids.tolist().index(sep_id)
            output_ids = output_ids[sep_index + 1:]
        else:
            print('sep token does not exist in output tokens')
    ja_case = case_en_ja[case]
    # output_tokens = tagger.parse(replace_t5tok_2_sptok(mega_tokenizer.decode(output_ids))).split()
    decoded_str = replace_t5tok_2_sptok(mega_tokenizer.decode(output_ids))
    decoded_str = re.sub(r'\s','',decoded_str)
    juman_result = jumanpp.analysis(decoded_str)
    output_tokens = [mrph.midasi for mrph in juman_result.mrph_list()]
    #「頼もしさ」の「さ」がガ格の項とき出力は「さが政府に求められる」となり解析不可？
    indices = [i for i, x in enumerate(output_tokens) if x == ja_case]
    if(len(indices) < 1):
        return ''
    else:
        return replace_sptok_2_t5tok(output_tokens[indices[0] - 1])
        #サブワードの末尾を取る場合はこのトークンをsentencepieceでトークナイズしてから末尾を取る
        #ガヲニを基準としてそこから前の部分を全部ラベルとする ex)党代表が雄弁に演説した　ガ格:党代表　or 代表,長めで学習してから評価で末尾のsubwordをとればよい

# %%
input_ids = mega_tokenizer.encode('私が食べる')
jumanpp.analysis('すもももももも')
# result = extrace_argument(input_ids,'ga') 

# %%
dataset = load_dataset('json',data_files={"test":'/home/sibava/PAS-T5/pas-dataset/pas_data.test.jsonl'})
input_dataset = dataset.with_format(type='torch',columns=['input_ids'])

# %%
tensor_dataloader = DataLoader(input_dataset['test'],batch_size=16)
pas_dataset = iter(dataset['test'])

# %%
with open('decoded_psa_juman.jsonl',mode='w') as f:
	pas_dataset = iter(dataset['test'])
	dataloader = DataLoader(dataset['test'],batch_size=16)
	for batch_input in tqdm(iter(tensor_dataloader)):
		outputs = mega_model.generate(batch_input['input_ids'].to(device))
		for output in outputs:
			psa_instance = next(pas_dataset)
			alt_type = psa_instance['alt_type']
			predicate = psa_instance['predicate']
			for case in ['ga','o','ni']:
				argument = extrace_argument(output,case)
				decode_dict = {
					'output_token':argument,
					'gold_arguments':psa_instance['gold_arguments'][case],
					'case_name':case,
					'case_type':psa_instance['case_types'][case],
					'predicate':predicate,
					'alt_type':alt_type,
					'input_tokens':psa_instance['input_tokens']
					}
				f.write(json.dumps(decode_dict) + '\n')


