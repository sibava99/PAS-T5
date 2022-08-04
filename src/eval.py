import json
import pandas as pd
from transformers import AutoTokenizer

# decode_dict = {
# 					'output':argument,
# 					'gold_arguments':psa_instance['gold_arguments'][case],
# 					'case_name':case,
# 					'arg_type':psa_instance['arg_types'][case],
# 					'predicate':predicate,
# 					'pred_sent_index':psa_instance['pred_sent_index'],
# 					'pred_indices':psa_instance['pred_indices'],
# 					'alt_type':alt_type,
# 					'output_sentence':t5_tokenizer.decode(output),
# 					'context':psa_instance['context'],
# 					'ntc_path':psa_instance['ntc_path']
# 					}

def print_scores(df_name:str,df):
	value_count = df['result'].value_counts()
	tp = value_count['tp']
	fp = value_count['fp']
	fn = value_count['fn']
	
	precision = tp/(tp + fp)
	recall = tp/(tp + fn + fp)
	
	f1 = 2*recall*precision/(recall + precision)
	print(f'DataFrame = {df_name}\nprecision : {precision} ({tp}/{tp + fp})\nrecall : {recall} ({tp}/{len(df)})\nf1 : {f1}\n')

def is_correct(output:str,gold_arguments:list,subword=False,no_distinct=False,bidirection=False)->bool:
	"""
	* subword : 評価の際にトークンをサブワード化した末尾が一致していれば正解とする
	* no_distinct : 評価の際に[#糸]と[糸]のようなサブワードトークンと単一トークンを区別しない
	"""
	output_token = output.translate(str.maketrans({chr(0x0021 + i): chr(0xFF01 + i) for i in range(94)})) #半角文字を全角文字に変換
	gold_arguments = [gold.translate(str.maketrans({chr(0x0021 + i): chr(0xFF01 + i) for i in range(94)})) for gold in gold_arguments]
	if(subword == True):
		output_token = bert_tokenizer.tokenize(output_token)[-1]
		gold_arguments = [bert_tokenizer.tokenize(argument)[-1] for argument in gold_arguments]
		if(no_distinct):
			output_token = output_token.replace('#','')
			gold_arguments = [argument.replace('#','') for argument in gold_arguments]

	if(bidirection):
		for gold in gold_arguments:
			if (output_token in gold or gold in output_token):
				result = True
			else:
				answer = False
	else:
		result = True if (output_token in gold_arguments) else False
	return answer


def make_dfdict(decoded_path:str) ->dict:
	f = open(decoded_path,mode='r')
	lines = f.readlines()
	for line in lines:
		psa_instance= json.loads(line)
		gold_arg_type = psa_instance['arg_type']
		if(psa_instance['output'] == ''):
			if(gold_arg_type == 'null'):
				continue
			else:
				results[gold_arg_type]['fn'].append(psa_instance)
		else:
			answer = is_correct(psa_instance['output'],psa_instance['gold_arguments'])

			if(answer):
				results[gold_arg_type]['tp'].append(psa_instance)
			else:
				df_psa.at[index,'result'] = 'fp'
	
	df_dict = {
		'df_dep' : df_psa.query('arg_type == "dep"'),
		'df_intra' : df_psa.query('arg_type == "intra"'),
		'df_inter' : df_psa.query('arg_type == "inter"'),
		'df_dep_passive' : df_psa.query('arg_type == "dep" and alt_type == "passive"'),
		'df_intra_passive' : df_psa.query('arg_type == "intra" and alt_type == "passive"'),
		'df_inter_passive' : df_psa.query('arg_type == "inter" and alt_type == "passive"'),
		'df_exo' : df_psa.query('arg_type == "exog" or arg_type == "exo1" or arg_type == "exo2"'),
		# 'df_zero': df_psa.query('case_type != "null" and case_type != "exog" and case_type != "exo1" and case_type != "exo2" and case_type != "dep"'),
		'df_zero': df_psa.query('arg_type != "null" and arg_type != "dep"'),
	}
	
	return df_dict


bert_tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
t5_tokenizer = AutoTokenizer.from_pretrained('megagonlabs/t5-base-japanese-web')

juman_dfdict = make_dfdict('/home/sibava/PAS-T5/decoded/decoded_psa_juman3.jsonl')
# mecab_dfdict = make_dfdict('/home/sibava/PAS-T5/decoded/decoded_psa_closest.jsonl')
mecab_dfdict = make_dfdict('/home/sibava/PAS-T5/decoded/suffix.jsonl')

mecab_dfdict['df_exo'].query('result == "fp"')[0:50]

df_juman_inter = juman_dfdict['df_inter']
df_juman_inter = df_juman_inter.rename(columns={"result":"juman_result","output_token":"juman_output_token"})
df_juman_inter = df_juman_inter.loc[:,['juman_output_token','output_sentence','juman_result']]


df_mecab_inter = mecab_dfdict['df_inter']

df_merge = df_mecab_inter.join(df_juman_inter)

for df_name,df in mecab_dfdict.items():
	print_scores(df_name,df)


