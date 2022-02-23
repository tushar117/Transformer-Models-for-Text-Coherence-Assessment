from utils.common import store_results
from logger import LOG_LEVELS, MyLogger, logger_wrapper
import json
from tqdm import tqdm
import os
import spacy as sp
import pandas as pd
import re


base_dir = os.path.dirname(os.path.realpath(__file__))

def load_file(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as dfile:
        for line in dfile.readlines():
            sentences.append(line.strip())
    return sentences

def process_sentences(file_path, max_sequence_length):
    sents = load_file(file_path)
    sents = [x for x in sents if x != '<para_break>' and x != '']

    if max_sequence_length == -1:
        return sents, False
    
    processed_sents = []
    word_length = 0
    is_truncated = False
    for x in sents:
        words = [w.text for w in sp(x)]
        if word_length + len(words) > max_sequence_length:
            is_truncated = True
            break
        word_length += len(words)
        processed_sents.append(x)
    return processed_sents, is_truncated

def prepare_wsj_factual_dataset(wsj_dataset_path, store_dataset_path, fact_extractor, logger=None):
    en_nlp = sp.load('en_core_web_sm')
    #create the store directory if not already exists
    os.makedirs(store_dataset_path, exist_ok=True)
    logger = logger_wrapper(logger, 'prepare_wsj_factual_dataset')
    set_splits = {
                'train': range(0, 11),
                'dev': range(10, 14),
                'test': range(14, 25),
                }
    
    for set_type in set_splits:
        res = []
        logger.info('creating %s dataset' % set_type)
        for folder_name in tqdm(set_splits[set_type]):
            for file_name in os.listdir(os.path.join(wsj_dataset_path, "%02d"%folder_name)):
                temp = {}
                file_path = os.path.join(wsj_dataset_path, "%02d"%folder_name, file_name)
                sents_list, is_truncated = process_sentences(file_path, -1)
                temp['sentences'] = [[y.text for y in  en_nlp(x)] for x in sents_list]
                temp['facts'] = []
                for sent in sents_list:
                    temp['facts'].append(fact_extractor.get_triples(sent))
                res.append(temp)
        store_results(os.path.join(store_dataset_path, '%s.jsonl'%set_type), res)

def prepare_gcdc_factual_dataset(gcdc_dataset_path, store_dataset_path, fact_extractor, logger=None):
    en_nlp = sp.load('en_core_web_sm')
    #create the store directory if not already exists
    os.makedirs(store_dataset_path, exist_ok=True)
    logger = logger_wrapper(logger, logger_name='prepare_gcdc_factual_dataset')
    
    for file_name in os.listdir(gcdc_dataset_path):
        if not file_name.endswith('.csv'):
            continue
        res = []
        logger.info('processing file: %s' % file_name)
        file_path = os.path.join(gcdc_dataset_path, file_name)
        dframe = pd.read_csv(file_path)
        for text_id, text, label, label1, label2, label3 in tqdm(zip(dframe['text_id'], dframe['text'], dframe['labelA'], dframe['ratingA1'], dframe['ratingA2'], dframe['ratingA3'])):
            temp = {
                'text_id': text_id,
                'text': text,
                'label1': label1,
                'label2': label2,
                'label3': label3,
                'label': label,
                }
            processed_text = re.sub(r"\s{1,}", " ", str(text))
            sents_list = [str(x.text).strip() for x in en_nlp(processed_text).sents]
            sents_list = [x for x in sents_list if x != '<para_break>' and x != '']
            temp['sentences'] = [[y.text for y in en_nlp(x)] for x in sents_list]
            temp['facts'] = []
            for sent in sents_list:
                temp['facts'].append(fact_extractor.get_triples(sent))
            res.append(temp)
        store_file_name = file_name.split('.', 1)[0]
        store_results(os.path.join(store_dataset_path, '%s.jsonl'%store_file_name), res)

def prepare_rte_dataset(rte_dataset_path, store_dataset_path, fact_extractor, logger=None):
    #create the store directory if not already exists
    os.makedirs(store_dataset_path, exist_ok=True)
    logger = logger_wrapper(logger, logger_name='prepare_rte_dataset')
    for file_name in os.listdir(rte_dataset_path):
        if not file_name.endswith('.tsv') or 'test' in file_name:
            continue
        logger.info('processing file: %s' % file_name)
        res = []
        dframe = pd.read_csv(os.path.join(rte_dataset_path, file_name), delimiter='\t')
        dframe = dframe.dropna()
        for text_id, sent1, sent2, label in zip(dframe['index'], dframe['sentence1'], dframe['sentence2'], dframe['label']):
            sent1 = re.sub(r"\s{1,}", " ", str(sent1))
            sent2 = re.sub(r"\s{1,}", " ", str(sent2))
            
            final_label = 0
            if label == 'entailment':
                final_label = 1
            
            res.append({
                'text_id': text_id,
                'doc_a': sent1,
                'doc_a_facts': fact_extractor.get_triples(sent1),
                'doc_b': sent2,
                'doc_b_facts': fact_extractor.get_triples(sent2),
                'label': final_label,
            })
        store_file_name = file_name.split('.', 1)[0]
        store_results(os.path.join(store_dataset_path, '%s.jsonl'%store_file_name), res)

def prepare_mnli_dataset(mnli_dataset_path, store_dataset_path, fact_extractor, logger=None):
    #create the store directory if not already exists
    os.makedirs(store_dataset_path, exist_ok=True)
    logger = logger_wrapper(logger, logger_name='prepare_mnli_dataset')
    file_map = {
        'dev': 'multinli_1.0_dev_matched.jsonl',
        'train': 'multinli_1.0_train.jsonl',
    }
    
    # specify the dataset count
    dataset_count = {
        'train': 21560,
        'dev': 7920,
    }

    for set_type, file_name in file_map.items():
        count = dataset_count[set_type]
        logger.info('processing file: %s (with count : %s)' % (file_name, count))
        res = []
        with open(os.path.abspath(os.path.join(mnli_dataset_path, file_name)), 'r', encoding='utf-8') as dfile:
            for line in dfile.readlines():
                item = json.loads(line.strip())
        
                if len(res) >= count:
                    break
                sent1 = item['sentence1'].strip()
                sent2 = item['sentence2'].strip()
                label = item['gold_label'].strip()

                # only include entailment and non-entailment data
                if label not in ['entailment', 'contradiction']:
                    continue
                
                final_label = 0
                if label == 'entailment':
                    final_label = 1
                
                res.append({
                    'doc_a': sent1,
                    'doc_a_facts': fact_extractor.get_triples(sent1),
                    'doc_b': sent2,
                    'doc_b_facts': fact_extractor.get_triples(sent2),
                    'label': final_label,
                })
        store_results(os.path.join(store_dataset_path, '%s.jsonl'%set_type), res)

def prepare_aes_data(dataset_dir, store_dataset_path, fact_extractor, logger=None):
    #create the store directory if not already exists
    os.makedirs(store_dataset_path, exist_ok=True)
    logger = logger_wrapper(logger, logger_name='prepare_aes_dataset')
    en_nlp = sp.load('en_core_web_sm')

    # Defining function to clean the essay
    def clean_text(text):
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text, flags=re.I)
        text = re.sub('<.*?>+', '', text)
        text = re.sub(r"\s{1,}", " ", text) 
        return text

    logger.info('loading the train essay scoring dataset from %s' % dataset_dir)
    with open(os.path.join(store_dataset_path, 'train.jsonl'), 'w', encoding='utf-8') as dfile:
        df_train = pd.read_csv(os.path.join(dataset_dir, 'training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')
        for essay_id, essay_set, essay, score in tqdm(zip(df_train['essay_id'], df_train['essay_set'], df_train['essay'], df_train['domain1_score'])):
            cleaned_essay = clean_text(essay)
            temp = {
                'esssay_id': essay_id,
                'essay_set': essay_set,
                'essay': essay,
                'score': score,
                'sentences': [x.text for x in en_nlp(essay).sents],
                'facts': [],
            }

            for s in temp['sentences']:
                temp['facts'].append(fact_extractor.get_triples(s))
            
            json.dump(temp, dfile)
            dfile.write('\n')