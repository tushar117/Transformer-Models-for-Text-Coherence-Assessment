import os, sys
import json
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter
import random
from utility import get_normalized_score, asap_score_ranges

# required to access the python modules present in project directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils.common import store_results, linearize_facts
from utils.data_specific import process_sentences_and_facts, featurize_data


def _featurize_sentence_ordering_dataset(args, set_type, dataset):
    random.seed(42)
    data_list = []
    # handling special ablation study cases
    mtl_with_fact_aware_model = (args.arch=='mtl' and args.mtl_base_arch=='fact-aware')
    mtl_with_hierarchical_model = (args.arch=='mtl' and args.mtl_base_arch=='hierarchical')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    fact_tokenizer = tokenizer
    # use TF2 for fact tokenizer when 'fact-aware' and 'combined' architecture is selected
    if args.arch in ['fact-aware', 'combined'] or mtl_with_fact_aware_model:
        fact_tokenizer = AutoTokenizer.from_pretrained(args.tf2_model_name)
    
    args.logger.info('featurizing the essay ranking corpus for task with model architecture: %s' % (args.arch))
    low_score, high_score = asap_score_ranges.get(args.prompt_id)
    
    include_facts = args.arch in ['fact-aware', 'combined'] or mtl_with_fact_aware_model
    sent_sep = tokenizer.sep_token if (args.arch in ['hierarchical', 'combined']) or mtl_with_hierarchical_model else ' '
    
    sentence_ordering_dataset_count = 0
    total_count = len(dataset)
    for i in range(total_count-1):
        doc1 = dataset[i]
        sample_count = min(total_count-i-1, args.nsamples)
        seen_examples = []
        for j in range(sample_count):
            rindx = random.randint(i+1, total_count - 1)
            while rindx in seen_examples:
                rindx = random.randint(i+1, total_count - 1)
            seen_examples.append(rindx)
            doc2 = dataset[rindx]

            doc1_essay_score = get_normalized_score(doc1['score'], low_score, high_score)
            doc2_essay_score = get_normalized_score(doc2['score'], low_score, high_score)
            if doc1_essay_score == doc2_essay_score:
                continue
            
            if doc1_essay_score < doc2_essay_score:
                # always make doc1 of higher score
                temp = doc1
                doc1 = doc2
                doc2 = temp

            if sentence_ordering_dataset_count%2==0:
                temp = {
                    'doc_a': doc1['essay'],
                    'doc_a_sentences': doc1['sentences'],
                    'doc_a_facts': doc1['facts'],
                    'doc_b': doc2['essay'],
                    'doc_b_sentences': doc2['sentences'],
                    'doc_b_facts': doc2['facts'],
                    'label': 1,
                }
            else:
                temp = {
                    'doc_a': doc2['essay'],
                    'doc_a_sentences': doc2['sentences'],
                    'doc_a_facts': doc2['facts'],
                    'doc_b': doc1['essay'],
                    'doc_b_sentences': doc1['sentences'],
                    'doc_b_facts': doc1['facts'],
                    'label': -1,
                }
            
            if args.arch in ['hierarchical', 'combined'] or mtl_with_hierarchical_model:
                temp['doc_a'] = sent_sep.join([' '.join(x) for x in temp['doc_a_sentences']])
                temp['doc_b'] = sent_sep.join([' '.join(x) for x in temp['doc_b_sentences']])
            data_list.append(temp)
            sentence_ordering_dataset_count+=1

    # integrating facts if required
    if include_facts:
        args.logger.debug('<<INTEGRATING>> factual information to essay evaluation dataset')
        args.logger.debug('working on %s dataset (count: %d)' % (set_type, len(data_list)))
        doc_a_truncated = 0
        doc_a_zero_fact = 0
        doc_b_truncated = 0
        doc_b_zero_fact = 0
        for data_instance in tqdm(data_list):
            doc_a_processed_sents, doc_a_processed_facts, doc_a_is_truncated, doc_a_is_zero_fact = process_sentences_and_facts(data_instance['doc_a_sentences'], data_instance['doc_a_facts'], args.max_seq_len) 
            doc_b_processed_sents, doc_b_processed_facts, doc_b_is_truncated, doc_b_is_zero_fact = process_sentences_and_facts(data_instance['doc_b_sentences'], data_instance['doc_b_facts'], args.max_seq_len)
            doc_a_truncated += 1 if doc_a_is_truncated else 0
            doc_a_zero_fact += 1 if doc_a_is_zero_fact else 0
            doc_b_truncated += 1 if doc_b_is_truncated else 0
            doc_b_zero_fact += 1 if doc_b_is_zero_fact else 0
            # processed sentence lists is required for creating sentence
            data_instance['doc_a'] = sent_sep.join([' '.join(x) for x in doc_a_processed_sents])
            data_instance['doc_b'] = sent_sep.join([' '.join(x) for x in doc_b_processed_sents])
            
            doc_a_fact_list = linearize_facts(doc_a_processed_facts)
            doc_b_fact_list = linearize_facts(doc_b_processed_facts)
            if args.max_fact_count>0:
                doc_a_fact_list = doc_a_fact_list[:args.max_fact_count]
                doc_b_fact_list = doc_b_fact_list[:args.max_fact_count]
            data_instance['doc_a_facts'] = doc_a_fact_list
            data_instance['doc_b_facts'] = doc_b_fact_list
        args.logger.debug('datapoints truncated | doc_a : %d, doc_b : %d (maximum seq length : %d)' % (doc_a_truncated, doc_b_truncated, args.max_seq_len))
        args.logger.debug('datapoints contains no facts | doc_a : %d, doc_b : %d out of %d document pair' % (doc_a_zero_fact, doc_b_zero_fact, len(data_list)))
        args.logger.debug('--'*30)
    
    final_data = featurize_data(tokenizer, fact_tokenizer, include_facts, args.max_fact_count, args.max_seq_len, args.max_fact_seq_len, data_list, args.logger)
    assert len(final_data) == len(data_list), "essay_scoring:featurize_dataset > datainstance count mismatch between featurized and actual dataset"
    # label distribution
    args.logger.info('label distribution in %s essay ranking dataset (total count: %d)' % (set_type, len(final_data)))
    lc = Counter([x['label'] for x in final_data])
    args.logger.info('%s' % ({k:v for k,v in lc.items()}))
    args.logger.debug('--'*30)
    return final_data

def _featurize_sentence_ordering_inference_data(args, set_type, dataset):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    fact_tokenizer = tokenizer
    data_list = []
    # handling special ablation study cases
    mtl_with_fact_aware_model = (args.arch=='mtl' and args.mtl_base_arch=='fact-aware')
    mtl_with_hierarchical_model = (args.arch=='mtl' and args.mtl_base_arch=='hierarchical')
    # use TF2 for fact tokenizer when 'fact-aware' and 'combined' architecture is selected
    if args.arch in ['fact-aware', 'combined'] or mtl_with_fact_aware_model:
        fact_tokenizer = AutoTokenizer.from_pretrained(args.tf2_model_name)
    
    args.logger.info('featurizing the inference essay ranking corpus for task with model architecture: %s' % (args.arch))
    
    include_facts = args.arch in ['fact-aware', 'combined'] or mtl_with_fact_aware_model
    sent_sep = tokenizer.sep_token if (args.arch in ['hierarchical', 'combined']) or mtl_with_hierarchical_model else ' '
    
    total_count = len(dataset)
    for i in range(total_count):
        doc1 = dataset[i]
        temp = {
            'essay_id': int(doc1['esssay_id']),
            'doc_a': doc1['essay'],
            'doc_a_sentences': doc1['sentences'],
            'doc_a_facts': doc1['facts'],
            'label': -1, # adding dummmy label for feature processing
        }
        if args.arch in ['hierarchical', 'combined'] or mtl_with_hierarchical_model:
            temp['doc_a'] = sent_sep.join([' '.join(x) for x in temp['doc_a_sentences']])
        data_list.append(temp)

    # integrating facts if required
    if include_facts:
        args.logger.debug('<<INTEGRATING>> factual information to essay evaluation dataset')
        args.logger.debug('working on %s dataset (count: %d)' % (set_type, len(data_list)))
        doc_a_truncated = 0
        doc_a_zero_fact = 0
        for data_instance in tqdm(data_list):
            doc_a_processed_sents, doc_a_processed_facts, doc_a_is_truncated, doc_a_is_zero_fact = process_sentences_and_facts(data_instance['doc_a_sentences'], data_instance['doc_a_facts'], args.max_seq_len) 
            doc_a_truncated += 1 if doc_a_is_truncated else 0
            doc_a_zero_fact += 1 if doc_a_is_zero_fact else 0
            # processed sentence lists is required for creating sentence
            data_instance['doc_a'] = sent_sep.join([' '.join(x) for x in doc_a_processed_sents])
            doc_a_fact_list = linearize_facts(doc_a_processed_facts)
            if args.max_fact_count>0:
                doc_a_fact_list = doc_a_fact_list[:args.max_fact_count]
            data_instance['doc_a_facts'] = doc_a_fact_list
        args.logger.debug('datapoints truncated | doc_a : %d (maximum seq length : %d)' % (doc_a_truncated, args.max_seq_len))
        args.logger.debug('datapoints contains no facts | doc_a : %d out of %d document pair' % (doc_a_zero_fact, len(data_list)))
        args.logger.debug('--'*30)
    
    final_data = featurize_data(tokenizer, fact_tokenizer, include_facts, args.max_fact_count, args.max_seq_len, args.max_fact_seq_len, data_list, args.logger)
    assert len(final_data) == len(data_list), "essay_scoring:featurize_dataset > datainstance count mismatch between featurized and actual dataset"
    for fd, dl in zip(final_data, data_list):
        fd['essay_id'] = dl['essay_id']
        # also remove the label
        del fd['label']
    return final_data

def load_essay_coherence_vector(args, fold_id):
    if args.coherence_vector:
        filename = os.path.abspath(args.coherence_vector)
        args.logger.critical("laoding coherence vectors from file : %s" % filename)
    else:
        filename = os.path.join(args.processed_dataset_path, "%s-coherence-vector-%d.json"%(args.arch, fold_id))
    with open(filename, 'r') as dfile:
        res = json.load(dfile)
    return res

def _featurize_dataset(args, set_type, dataset, fold_id):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    data_list = []
    args.logger.info('featurizing the essay scoring corpus for task with model architecture: vanilla')
    low_score, high_score = asap_score_ranges.get(args.prompt_id)
    
    if args.enable_coherence_signal:
        id_2_coherence_vector = load_essay_coherence_vector(args, fold_id)

    for data_item in dataset:
        temp = {
            'essay_id': data_item['esssay_id'],
            'prompt_id': data_item['essay_set'],
            'doc_a': data_item['essay'],
            'label': get_normalized_score(data_item['score'], low_score, high_score),
        }
        data_list.append(temp)

    final_data = featurize_data(tokenizer, tokenizer, False, args.max_fact_count, args.max_seq_len, args.max_fact_seq_len, data_list, args.logger)
    assert len(final_data) == len(data_list), "essay_scoring:featurize_dataset > datainstance count mismatch between featurized and actual dataset"
    for fd, dl in zip(final_data, data_list):
        fd['prompt_id'] = dl['prompt_id']
        if args.enable_coherence_signal:
            # adding the coherence signal
            fd['coherence_vector'] = id_2_coherence_vector[str(dl['essay_id'])]
    
    return final_data

def featurize_train_dataset(args, set_type, dataset, fold_id, sentence_ordering=False):
    if sentence_ordering:
        res = _featurize_sentence_ordering_dataset(args, set_type, dataset)
    else:
        res = _featurize_dataset(args, set_type, dataset, fold_id)

    save_dir = os.path.join(args.processed_dataset_path, 'featurized_dataset')
    os.makedirs(save_dir, exist_ok=True)
    save_file_path = os.path.join(save_dir, '%s.jsonl'%set_type)
    store_results(save_file_path, res)

def featurize_test_dataset(args, dataset, fold_id, sentence_ordering=False):
    if sentence_ordering:
        res = _featurize_sentence_ordering_inference_data(args, 'test', dataset)
    else:
        res = _featurize_dataset(args, 'test', dataset, fold_id)
    save_dir = os.path.join(args.processed_dataset_path, 'featurized_dataset')
    os.makedirs(save_dir, exist_ok=True)
    save_file_path = os.path.join(save_dir, 'test.jsonl')
    store_results(save_file_path, res)

