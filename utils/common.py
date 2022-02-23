import json
import os
from tqdm import tqdm
from collections import defaultdict
import shutil
import random
import numpy as np


def random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def load_file(file_path):
    res = []
    with open(os.path.abspath(file_path), 'r', encoding='utf-8') as dfile:
        for line in dfile.readlines():
            res.append(json.loads(line.strip()))
    return res

def clean_directory(dir_path): 
    for the_file in os.listdir(dir_path):
        #remove files related to dataset creation
        if not the_file.endswith('.jsonl'):
            continue
        file_path = os.path.join(dir_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print("unable to delete path %s. error : %s" % (the_file, str(e)))

def full_clean_directory(dir_path): 
    for the_file in os.listdir(dir_path):
        #remove files related to dataset creation
        file_path = os.path.join(dir_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print("unable to delete path %s. error : %s" % (the_file, str(e)))

def store_results(save_file_path, res):
    with open(save_file_path, 'w', encoding='utf-8') as dfile:
        for item in res:
            json.dump(item, dfile)
            dfile.write('\n')

def normalize_gcdc_sub_corpus(corpus_str):
    corpus_str = corpus_str.strip()
    normalized_sub_corpus = "%s%s"%(corpus_str[0].upper(), corpus_str[1:].lower())
    return normalized_sub_corpus

def encode_batch(tokenizer, data_list, max_seq_len):
    tokenzier_args = {'batch_text_or_text_pairs': data_list, 'padding': 'longest', 'return_attention_mask': True}
    if max_seq_len>0:
        tokenzier_args.update({'padding': 'max_length', 'max_length': max_seq_len, 'truncation': True})
    
    tokenized_data = tokenizer.batch_encode_plus(**tokenzier_args)
    return tokenized_data['input_ids'], tokenized_data['attention_mask']

def linearize_facts(sent_facts):
    facts = []
    for x in sent_facts:
        #also removes the empty facts 
        facts.extend(x)
    return facts