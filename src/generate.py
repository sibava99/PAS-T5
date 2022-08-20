import json
import pprint
import torch
import re
from tqdm import tqdm
import argparse
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


def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model','-m',help='path to model')
	parser.add_argument('--test','-t',help='test data generated by PSA2T5.py')
	parser.add_argument('--output','-o',default='decode.jsonl',help='output file path')
	return parser

def get_index(l, x, default=False):
    return l.index(x) if x in l else -256

def extract_argument(output_ids:list,case:str,is_alt:bool=False) -> str:
    sep_id = t5_tokenizer.encode('<extra_id_2>',add_special_tokens=None)[0]
    if(is_alt):
        if(sep_id in output_ids.tolist()):
            sep_index = output_ids.tolist().index(sep_id)
            output_ids = output_ids[sep_index + 1:]
        else:
            print('sep token does not exist in output tokens')
    decoded_str = t5_tokenizer.decode(output_ids)
    decoded_str = re.sub(r'\s|<pad>','',decoded_str).translate(str.maketrans({chr(0x0021 + i): chr(0xFF01 + i) for i in range(94)}))#半角→全角
    ga_index = get_index(decoded_str,'が')
    o_index = get_index(decoded_str,'を')
    ni_index = get_index(decoded_str,'に')
    
    ja_case = case_en_ja[case]
    if(case == 'ga'):
        if(ga_index == -256):
            return ''
        else:
            return decoded_str[:ga_index]
    elif(case == 'o'):
        if(o_index == -256):
            return ''
        elif(ga_index == -256):
            return decoded_str[:o_index]
        else:
            return decoded_str[ga_index+1:o_index]
    elif(case == 'ni'):
        if(ni_index == -256):
            return ''
        elif(o_index != -256):
            return decoded_str[o_index+1:ni_index]
        elif(o_index == -256 and ga_index != 0):
            return decoded_str[ga_index + 1:ni_index]
        else:
            return decoded_str[0:ni_index]
    else:
        return 
        #サブワードの末尾を取る場合はこのトークンをsentencepieceでトークナイズしてから末尾を取る
        #ガヲニを基準としてそこから前の部分を全部ラベルとする ex)党代表が雄弁に演説した　ガ格:党代表　or 代表,長めで学習してから評価で末尾のsubwordをとればよい

parser = create_parser()
args = parser.parse_args()
device = 'cuda:0'

t5_tokenizer = T5Tokenizer.from_pretrained("megagonlabs/t5-base-japanese-web")
t5_model = T5ForConditionalGeneration.from_pretrained("megagonlabs/t5-base-japanese-web").to(device)
t5_model.resize_token_embeddings(len(t5_tokenizer))
t5_model.load_state_dict(torch.load(args.model))

t5_model.eval()

case_en_ja = {
	'ga':'が',
	'o':'を',
	'ni':'に',
	'yotte':'によって'
}

dataset = load_dataset('json',data_files={"test":args.test})
input_dataset = dataset.with_format(type='torch',columns=['input_ids'])

tensor_dataloader = DataLoader(input_dataset['test'],batch_size=16)
pas_dataset = iter(dataset['test'])

with open(args.output,mode='w') as f:
	pas_dataset = iter(dataset['test'])
	dataloader = DataLoader(dataset['test'],batch_size=16)
	for batch_input in tqdm(iter(tensor_dataloader)):
		outputs = t5_model.generate(batch_input['input_ids'].to(device))
		for output in outputs:
			psa_instance = next(pas_dataset)
			alt_type = psa_instance['alt_type']
			predicate = psa_instance['predicate']
			for case in ['ga','o','ni']:
				if (psa_instance['arg_types'][case] == None):
					continue
				argument = extract_argument(output,case)
				# argument = t5_tokenizer.decode(output,skip_special_tokens=True)
				decode_dict = {
					'output':argument,
					'gold_arguments':psa_instance['gold_arguments'][case],
					'case_name':case,
					'arg_type':psa_instance['arg_types'][case],
					'predicate':predicate,
					'pred_sent_index':psa_instance['pred_sent_index'],
					'pred_indices':psa_instance['pred_indices'],
					'pred_bunsetsu_index':psa_instance['pred_bunsetsu_index'],
					'alt_type':alt_type,
					'output_sentence':t5_tokenizer.decode(output),
					'context':psa_instance['context'],
					'ntc_path':psa_instance['ntc_path']
					}
				f.write(json.dumps(decode_dict) + '\n')


