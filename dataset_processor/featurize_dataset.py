import json
import os
from utils.common import load_file, store_results, normalize_gcdc_sub_corpus, encode_batch, linearize_facts, clean_directory
from utils.data_specific import process_sentences_and_facts, featurize_dataset, featurize_data, get_permutated_sentence_pairs
from collections import Counter
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import Counter
import random


#defines valid task that can be performed on the dataset
dataset_options = {
    "wsj": {
        "tasks" : ["sentence-ordering"],
    },
    "gcdc": {
        "tasks" : ['3-way-classification', 'minority-classification', 'sentence-ordering', 'sentence-score-prediction'],
        "sub_corpus" : ['All', 'Clinton', 'Enron', 'Yelp', 'Yahoo'],
    }
}

class GCDCFeaturizer:
    def __init__(self, args):
        if args.task not in dataset_options['gcdc']['tasks']:
            raise Exception('handler for %s task is not defined.' % (args.task))
        if args.sub_corpus.lower() not in [x.lower() for x in dataset_options['gcdc']['sub_corpus']]:
            raise Exception('%s is not part of GCDC corpus' % (args.sub_corpus))

        args.logger.info('featurizing the gcdc corpus: %s for task: %s with model architecture: %s' % (args.sub_corpus, args.task, args.arch))
        #populating the corpus list
        self.corpus_list = []
        if args.sub_corpus.lower() == 'all':
            self.corpus_list = dataset_options['gcdc']['sub_corpus'][1:]
        else:
            self.corpus_list = [normalize_gcdc_sub_corpus(args.sub_corpus)]
        #cache the cmdline arguments
        self.args = args
    
    def get_task_specific_labels(self, data_input_map):
        if self.args.task == 'sentence-score-prediction':
            annotator_aggreement = [data_input_map['label1'], data_input_map['label2'], data_input_map['label3']]
            annotator_aggreement = [x for x in map(lambda x: int(x), annotator_aggreement)]
            #regression task and it can be in range [0, 3]
            labels = np.mean(annotator_aggreement)
            return labels
        elif self.args.task == 'minority-classification':
            annotator_aggreement = [data_input_map['label1'], data_input_map['label2'], data_input_map['label3']]
            annotator_aggreement = [x for x in map(lambda x: int(x), annotator_aggreement)]
            cc = Counter(annotator_aggreement)
            label, freq = cc.most_common(1)[0]
            #binary classification task, where it takes value either zero (denotes non-low coherence text) or one (denotes low coherence text).
            labels=0 # not low coherence
            if label==1 and freq >= 2:
                labels=1 # low coherence
            return labels
        elif self.args.task == '3-way-classification':
            # its 3-class classification problem
            labels = int(data_input_map['label']) - 1    # actual gcdc labels are {1=low, 2=medium, 3=high}, transformed to [0=low, 1=medium, 2=high].  
            return labels
        else:
            raise Exception('%s task specific label processing is not defined' % self.args.task)
    
    def load_data(self, inference):
        dataset_map = {
            'train': [],
            'dev': [],
            'test': []
        }
        
        if inference:
            # removing training and development dataset while doing inference
            del dataset_map['dev']
            del dataset_map['train']
        else:
            del dataset_map['test']

        gcdc_dataset_path = os.path.join(self.args.processed_dataset_path, 'GCDC')
        self.args.logger.debug('<<LOADING>> GCDC dataset from directory: %s' % gcdc_dataset_path)
        for set_type in dataset_map:
            self.args.logger.debug('working on %s dataset ' % set_type)
            for sub_corpus in self.corpus_list:
                file_name = "%s_%s.jsonl"%(sub_corpus, set_type)
                input_data = load_file(os.path.join(gcdc_dataset_path, file_name))
                dataset_map[set_type].extend(input_data)
            self.args.logger.debug('<Done>')
        self.args.logger.debug('--'*30)
        return dataset_map

    def featurize_dataset(self, inference=False):
        # handling special cases for the ablation study cases
        mtl_with_fact_aware_model = (self.args.arch=='mtl' and self.args.mtl_base_arch=='fact-aware')
        mtl_with_hierarchical_model = (self.args.arch=='mtl' and self.args.mtl_base_arch=='hierarchical')
        
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        fact_tokenizer = tokenizer
        # use TF2 for fact tokenizer when 'fact-aware' and 'combined' architecture is selected
        if self.args.arch in ['fact-aware', 'combined'] or mtl_with_fact_aware_model:
            fact_tokenizer = AutoTokenizer.from_pretrained(self.args.tf2_model_name)
        save_dir = os.path.join(self.args.processed_dataset_path, 'featurized_dataset')
        os.makedirs(save_dir, exist_ok=True)
        dataset_map = self.load_data(inference)
        include_facts = (self.args.arch in ['fact-aware', 'combined']) or mtl_with_fact_aware_model
        sent_sep = tokenizer.sep_token if (self.args.arch in ['hierarchical', 'combined']) or mtl_with_hierarchical_model else ' '

        # task specific handler
        if self.args.task in ['3-way-classification', 'minority-classification', 'sentence-score-prediction']: 
            for set_type, data_list in dataset_map.items():
                for data_instance in data_list:
                    if self.args.arch in ['hierarchical', 'combined'] or mtl_with_hierarchical_model:
                        data_instance['doc_a'] = sent_sep.join([' '.join(x) for x in data_instance['sentences']])
                    else:
                        data_instance['doc_a'] = data_instance['text']
                    data_instance['label'] = self.get_task_specific_labels(data_instance)

        # integrating facts if required
        if include_facts:
            self.args.logger.debug('<<INTEGRATING>> factual information to GCDC dataset')
            for set_type, data_list in dataset_map.items():
                self.args.logger.debug('working on %s dataset (count: %d)' % (set_type, len(data_list)))
                truncated = 0
                zero_fact = 0
                for data_instance in tqdm(data_list):
                    processed_sents, processed_facts, is_truncated, is_zero_fact = process_sentences_and_facts(data_instance['sentences'], data_instance['facts'], self.args.max_seq_len) 
                    truncated += 1 if is_truncated else 0
                    zero_fact += 1 if is_zero_fact else 0
                    # processed sentence lists is required for creating sentence
                    if self.args.task == 'sentence-ordering':
                        data_instance['sentences'] = processed_sents
                        data_instance['facts'] = processed_facts
                    else:
                        data_instance['doc_a'] = sent_sep.join([' '.join(x) for x in processed_sents])
                        fact_list = linearize_facts(processed_facts)
                        if self.args.max_fact_count>0:
                            fact_list = fact_list[:self.args.max_fact_count]
                        data_instance['doc_a_facts'] = fact_list
                self.args.logger.debug('%d datapoints truncated (maximum seq length : %d)' % (truncated, self.args.max_seq_len))
                self.args.logger.debug('%d / %d datapoints contains no facts' % (zero_fact, len(data_list)))
            self.args.logger.debug('--'*30)

        # pairwise sentence ordering task handler 
        if self.args.task == "sentence-ordering":
            for set_type, data_list in dataset_map.items():
                
                permutated_pair_data_list = get_permutated_sentence_pairs(data_list, include_facts, self.args.permutation_count, 
                                                    self.args.logger, with_replacement=self.args.with_replacement>0,
                                                    sentence_separator=sent_sep, inverse=self.args.inverse_pra>0)
                #update the dataset with permutated sentence pair
                dataset_map[set_type] = permutated_pair_data_list

        # featurizing the dataset
        for set_type, data_list in dataset_map.items():
            save_file_path = os.path.join(save_dir, '%s.jsonl'%set_type)
            featurize_dataset(tokenizer, fact_tokenizer, include_facts, self.args.max_fact_count, self.args.max_seq_len, 
                                            self.args.max_fact_seq_len, save_file_path, data_list, self.args.logger)
        self.args.logger.debug('--'*30)
        
        # label distribution
        if self.args.task != 'sentence-score-prediction':
            for set_type, data_list in dataset_map.items():
                self.args.logger.info('label distribution in %s dataset (total count: %d)' % (set_type, len(data_list)))
                lc = Counter([x['label'] for x in data_list])
                self.args.logger.info('%s' % ({k:v for k,v in lc.items()}))
            self.args.logger.debug('--'*30)


class WSJFeaturizer:
    def __init__(self, args):
        if args.task not in dataset_options['wsj']['tasks']:
            raise Exception('handler for %s task is not defined.' % (args.task))

        args.logger.info('featurizing the wsj corpus for task: %s with model architecture: %s' % (args.task, args.arch))
        #cache the cmdline arguments
        self.args = args
    
    def load_data(self, inference):
        dataset_map = {
            'train': [],
            'dev': [],
            'test': []
        }

        if inference:
            # removing training and development dataset while doing inference
            del dataset_map['dev']
            del dataset_map['train']
        else:
            del dataset_map['test']

        wsj_dataset_path = os.path.join(self.args.processed_dataset_path, 'WSJ')
        self.args.logger.debug('<<LOADING>> WSJ dataset from directory: %s' % wsj_dataset_path)
        for set_type in dataset_map:
            self.args.logger.debug('working on %s dataset ' % set_type)
            file_name = "%s.jsonl"%(set_type)
            input_data = load_file(os.path.join(wsj_dataset_path, file_name))
            dataset_map[set_type] = input_data
            self.args.logger.debug('<Done>')
        self.args.logger.debug('--'*30)
        return dataset_map

    def featurize_dataset(self, inference=False):
        # handling special ablation study cases
        mtl_with_fact_aware_model = (self.args.arch=='mtl' and self.args.mtl_base_arch=='fact-aware')
        mtl_with_hierarchical_model = (self.args.arch=='mtl' and self.args.mtl_base_arch=='hierarchical')

        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        fact_tokenizer = tokenizer
        # use TF2 for fact tokenizer when 'fact-aware' and 'combined' architecture is selected
        if self.args.arch in ['fact-aware', 'combined'] or mtl_with_fact_aware_model:
            fact_tokenizer = AutoTokenizer.from_pretrained(self.args.tf2_model_name)
        save_dir = os.path.join(self.args.processed_dataset_path, 'featurized_dataset')
        os.makedirs(save_dir, exist_ok=True)
        dataset_map = self.load_data(inference)
        include_facts = self.args.arch in ['fact-aware', 'combined'] or mtl_with_fact_aware_model
        sent_sep = tokenizer.sep_token if (self.args.arch in ['hierarchical', 'combined']) or mtl_with_hierarchical_model else ' '

        # integrating facts if required
        if include_facts:
            self.args.logger.debug('<<INTEGRATING>> factual information to WSJ dataset')
            for set_type, data_list in dataset_map.items():
                self.args.logger.debug('working on %s dataset (count: %d)' % (set_type, len(data_list)))
                truncated = 0
                zero_fact = 0
                for data_instance in tqdm(data_list):
                    processed_sents, processed_facts, is_truncated, is_zero_fact = process_sentences_and_facts(data_instance['sentences'], data_instance['facts'], self.args.max_seq_len) 
                    truncated += 1 if is_truncated else 0
                    zero_fact += 1 if is_zero_fact else 0
                    # processed sentence lists is required for creating sentence
                    data_instance['sentences'] = processed_sents
                    data_instance['facts'] = processed_facts
                self.args.logger.debug('%d datapoints truncated (maximum seq length : %d)' % (truncated, self.args.max_seq_len))
                self.args.logger.debug('%d / %d datapoints contains no facts' % (zero_fact, len(data_list)))
            self.args.logger.debug('--'*30)

        # pairwise sentence ordering task handler 
        for set_type, data_list in dataset_map.items(): 
            permutated_pair_data_list = get_permutated_sentence_pairs(data_list, include_facts, self.args.permutation_count, 
                                                self.args.logger, with_replacement=self.args.with_replacement>0,
                                                sentence_separator=sent_sep, inverse=self.args.inverse_pra>0)
            #update the dataset with permutated sentence pair
            dataset_map[set_type] = permutated_pair_data_list

        # featurizing the dataset
        for set_type, data_list in dataset_map.items():
            save_file_path = os.path.join(save_dir, '%s.jsonl'%set_type)
            featurize_dataset(tokenizer, fact_tokenizer, include_facts, self.args.max_fact_count, self.args.max_seq_len, 
                                            self.args.max_fact_seq_len, save_file_path, data_list, self.args.logger)
        self.args.logger.debug('--'*30)
        
        # label distribution
        for set_type, data_list in dataset_map.items():
            self.args.logger.info('label distribution in %s dataset (total count: %d)' % (set_type, len(data_list)))
            lc = Counter([x['label'] for x in data_list])
            self.args.logger.info('%s' % ({k:v for k,v in lc.items()}))
        self.args.logger.debug('--'*30)

class MTLFeaturizer:
    def __init__(self, args, **kwargs):
        args.logger.info('featurizing the textual entailment corpus for task: %s with model architecture: %s' % (args.task, args.arch))
        #cache the cmdline arguments
        self.args = args
        self.kwargs = kwargs
    
    def load_data(self):
        dataset_map = {
            'train': [],
            'dev': [],
        }

        # use MNLI dataset whenever WSJ dataset is selected
        textual_entailment_dataset = 'MNLI' if self.args.corpus == 'wsj' else 'RTE'
        te_dataset_path = os.path.join(self.args.processed_dataset_path, textual_entailment_dataset)
        self.args.logger.debug('<<LOADING>> %s dataset from directory: %s' % (textual_entailment_dataset, te_dataset_path))
        for set_type in dataset_map:
            self.args.logger.debug('working on %s dataset ' % set_type)
            file_name = "%s.jsonl"%(set_type)
            input_data = load_file(os.path.join(te_dataset_path, file_name))
            dataset_map[set_type] = input_data
            self.args.logger.debug('<Done>')
        self.args.logger.debug('--'*30)
        return dataset_map
    
    def featurize_dataset(self, inference=False):
        # handling special cases for ablation study
        include_facts = (self.args.arch=='mtl' and self.args.mtl_base_arch=='fact-aware')

        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        fact_tokenizer = tokenizer
        # use TF2 for fact tokenizer when 'fact-aware' and 'combined' architecture is selected
        if self.args.arch in ['fact-aware', 'combined'] or include_facts:
            fact_tokenizer = AutoTokenizer.from_pretrained(self.args.tf2_model_name)
        save_dir = os.path.join(self.args.processed_dataset_path, 'featurized_dataset')
        os.makedirs(save_dir, exist_ok=True)
        te_dataset_map = self.load_data()
        
        set_types = ['test'] if inference else ['dev', 'train']

        mtl_featurized_dataset_map = {k:[] for k in set_types }

        #load the previously featurized dataset
        for file_prefix in mtl_featurized_dataset_map:
            file_name = "%s.jsonl"%file_prefix
            featurized_input_data = load_file(os.path.join(save_dir, file_name))
            # adding dataset identifier for the present task
            for item in featurized_input_data:
                item['d_id'] = 1
            mtl_featurized_dataset_map[file_prefix] = featurized_input_data
        
        # check if processed dataset is for sentence ordering 
        sentence_ordering_flag = 'doc_b' in mtl_featurized_dataset_map[random.choice(set_types)][0]
        
        # remove previously obtained dataset
        clean_directory(save_dir)
        
        te_featurized_dataset_map = {}
        
        # don't include Textual entailment dataset at inference time
        if not inference:
            te_featurized_dataset_map = {
                'train': [],
                'dev': [],
            }

            # featurize the Textual entailment dataset
            for set_type in te_featurized_dataset_map:
                # fetch the max_seq_len from previously featurized task specific dataset
                prev_max_seq_len = len(mtl_featurized_dataset_map[set_type][0]['doc_a'])
                prev_max_fact_count = self.args.max_fact_count
                prev_max_fact_seq_len = self.args.max_fact_seq_len
                # change according to the base architecture used by mtl
                if include_facts:
                    prev_max_fact_count = len(mtl_featurized_dataset_map[set_type][0]['doc_a_facts'])
                    prev_max_fact_seq_len = len(mtl_featurized_dataset_map[set_type][0]['doc_a_facts'][0])
                data_instances = te_dataset_map[set_type]
                
                temp = []
                for item in data_instances:
                    raw_data = {
                        # concatenating the hypothesis and premise separated with [SEP] token
                        'doc_a': item['doc_a'] + tokenizer.sep_token + item['doc_b'],
                        'label': item['label'],
                    }
                    if include_facts:
                        raw_data.update({'doc_a_facts': item['doc_a_facts'] + item['doc_b_facts']})
                    if sentence_ordering_flag:
                        # appending dummy doc_b
                        raw_data.update({
                            'doc_b': ' ',
                        })
                        if include_facts:
                            raw_data.update({'doc_b_facts': []})
    
                    temp.append(raw_data)
                res = featurize_data(tokenizer, fact_tokenizer, include_facts, prev_max_fact_count, prev_max_seq_len, prev_max_fact_seq_len, temp, self.args.logger, exact_count=True)
                for item in res:
                    item['d_id'] = 0
                    if 'add_entry' in self.kwargs:
                        item.update(self.kwargs['add_entry'])
                te_featurized_dataset_map[set_type] = res

        random.seed(42)
        # merge the task specific and textual entailment dataset
        for set_type in mtl_featurized_dataset_map:
            if set_type in te_featurized_dataset_map:
                mtl_featurized_dataset_map[set_type].extend(te_featurized_dataset_map[set_type])
            # mixing in different datasets
            random.shuffle(mtl_featurized_dataset_map[set_type])
            save_file_path = os.path.join(save_dir, '%s.jsonl'%set_type)
            store_results(save_file_path, mtl_featurized_dataset_map[set_type])

        self.args.logger.debug('--'*30)
        
        # label distribution
        for set_type, data_list in te_featurized_dataset_map.items():
            self.args.logger.info('label distribution for %s textual entailment dataset (total count: %d)' % (set_type, len(data_list)))
            lc = Counter([x['label'] for x in data_list if x['d_id'] == 0])
            self.args.logger.info('%s' % ({k:v for k,v in lc.items()}))
        self.args.logger.debug('--'*30)

class CombinedFeaturizer:
    def __init__(self, args, **kwargs):
        args.logger.info('featurizing the textual entailment corpus for task: %s with model architecture: %s' % (args.task, args.arch))
        #cache the cmdline arguments
        self.args = args
        self.kwargs = kwargs
    
    def load_data(self):
        dataset_map = {
            'train': [],
            'dev': [],
        }

        # use MNLI dataset whenever WSJ dataset is selected
        textual_entailment_dataset = 'MNLI' if self.args.corpus == 'wsj' else 'RTE'
        te_dataset_path = os.path.join(self.args.processed_dataset_path, textual_entailment_dataset)
        self.args.logger.debug('<<LOADING>> %s dataset from directory: %s' % (textual_entailment_dataset, te_dataset_path))
        for set_type in dataset_map:
            self.args.logger.debug('working on %s dataset ' % set_type)
            file_name = "%s.jsonl"%(set_type)
            input_data = load_file(os.path.join(te_dataset_path, file_name))
            dataset_map[set_type] = input_data
            self.args.logger.debug('<Done>')
        self.args.logger.debug('--'*30)
        return dataset_map
    
    def featurize_dataset(self, inference=False):
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        fact_tokenizer = tokenizer
        # use TF2 for fact tokenizer when 'fact-aware' and 'combined' architecture is selected
        if self.args.arch in ['fact-aware', 'combined']:
            fact_tokenizer = AutoTokenizer.from_pretrained(self.args.tf2_model_name)
        save_dir = os.path.join(self.args.processed_dataset_path, 'featurized_dataset')
        os.makedirs(save_dir, exist_ok=True)
        te_dataset_map = self.load_data()
        
        set_types = ['test'] if inference else ['dev', 'train']

        mtl_featurized_dataset_map = {k:[] for k in set_types }

        #load the previously featurized dataset
        for file_prefix in mtl_featurized_dataset_map:
            file_name = "%s.jsonl"%file_prefix
            featurized_input_data = load_file(os.path.join(save_dir, file_name))
            # adding dataset identifier for the present task
            for item in featurized_input_data:
                item['d_id'] = 1
            mtl_featurized_dataset_map[file_prefix] = featurized_input_data
        
        # check if processed dataset is for sentence ordering 
        sentence_ordering_flag = 'doc_b' in mtl_featurized_dataset_map[random.choice(set_types)][0]
        
        # remove previously obtained dataset
        clean_directory(save_dir)
        
        te_featurized_dataset_map = {}
        
        # don't include Textual entailment dataset at inference time
        if not inference and self.args.disable_mtl<=0:
            te_featurized_dataset_map = {
                'train': [],
                'dev': [],
            }

            # featurize the Textual entailment dataset
            for set_type in te_featurized_dataset_map:
                # fetch the max sequence information from previously featurized task specific dataset
                prev_max_seq_len = len(mtl_featurized_dataset_map[set_type][0]['doc_a'])
                prev_max_fact_count = len(mtl_featurized_dataset_map[set_type][0]['doc_a_facts'])
                prev_max_fact_seq_len = len(mtl_featurized_dataset_map[set_type][0]['doc_a_facts'][0])
                data_instances = te_dataset_map[set_type]
                
                temp = []
                for item in data_instances:
                    raw_data = {
                        # concatenating the hypothesis and premise separated with [SEP] token
                        'doc_a': item['doc_a'] + tokenizer.sep_token + item['doc_b'],
                        'label': item['label'],
                        'doc_a_facts': item['doc_a_facts'] + item['doc_b_facts'],
                    }
                    if sentence_ordering_flag:
                        # appending dummy doc_b
                        raw_data.update({
                            'doc_b': ' ',
                            'doc_b_facts': [],
                        })
                    temp.append(raw_data)
                res = featurize_data(tokenizer, fact_tokenizer, True, prev_max_fact_count, prev_max_seq_len, prev_max_fact_seq_len, temp, self.args.logger, exact_count=True)
                for item in res:
                    item['d_id'] = 0
                    if 'add_entry' in self.kwargs:
                        item.update(self.kwargs['add_entry'])
                te_featurized_dataset_map[set_type] = res

        random.seed(42)
        # merge the task specific and textual entailment dataset
        for set_type in mtl_featurized_dataset_map:
            if set_type in te_featurized_dataset_map:
                mtl_featurized_dataset_map[set_type].extend(te_featurized_dataset_map[set_type])
            # mixing in different datasets
            random.shuffle(mtl_featurized_dataset_map[set_type])
            save_file_path = os.path.join(save_dir, '%s.jsonl'%set_type)
            store_results(save_file_path, mtl_featurized_dataset_map[set_type])

        self.args.logger.debug('--'*30)
        
        # label distribution
        for set_type, data_list in te_featurized_dataset_map.items():
            self.args.logger.info('label distribution for %s textual entailment dataset (total count: %d)' % (set_type, len(data_list)))
            lc = Counter([x['label'] for x in data_list if x['d_id'] == 0])
            self.args.logger.info('%s' % ({k:v for k,v in lc.items()}))
        self.args.logger.debug('--'*30)