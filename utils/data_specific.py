import numpy as np
import itertools
from .common import random_seed, linearize_facts, store_results, encode_batch
from tqdm import tqdm


def get_permutations(sentence_count, permutation_count):
    res = []
    if sentence_count < 6:
        total_count = 0
        original_order = [x for x in range(sentence_count)]
        perms = set(itertools.permutations(original_order))
        for perm_order in perms:
            perm_order = list(perm_order)
            if np.all(np.array(perm_order)==np.array(original_order)):
                continue
            if total_count >= permutation_count:
                break
            total_count+=1
            res.append(perm_order)
    else:
        original_order = np.array([x for x in range(sentence_count)])
        prev_perms = []
        for j in range(permutation_count):
            perm_order = np.random.permutation(sentence_count)
            perm_str = ','.join([str(z) for z in perm_order])
            while np.all(perm_order==original_order) or perm_str in prev_perms:
                perm_order = np.random.permutation(sentence_count)
                perm_str = ','.join([str(z) for z in perm_order])
            prev_perms.append(perm_str)
            res.append(perm_order)
    random.shuffle(res)
    return res

def get_permutations_with_replacement(sentence_count, permutation_count):
    res = []
    original_order = np.array([x for x in range(sentence_count)])
    for j in range(permutation_count):
        perm_order = np.random.permutation(sentence_count)
    
        while np.all(perm_order==original_order):
            perm_order = np.random.permutation(sentence_count)
        res.append(perm_order)
    return res

def get_permutated_sentence_pairs(data_list, include_facts, permutations, logger, seed=42, with_replacement=True, inverse=False, sentence_separator=' '): 
    random_seed(seed)
    res = []
    one_sent_doc_count = 0
    total_count = 0

    for data in tqdm(data_list):
        total_count += 1
        #remove documents containing only one sentence
        sentences = data['sentences']
        
        if len(sentences)<=1:
            one_sent_doc_count+=1
            continue
        
        if include_facts:
            p_facts = data['facts']
            assert len(sentences) == len(p_facts), "mismatch in count of sentences and facts"

        if with_replacement:
            permutation_ordering = get_permutations_with_replacement(len(sentences), permutations)
        else:
            permutation_ordering = get_permutations(len(sentences), permutations)

        if inverse:
            permutation_ordering = [[len(sentences)-1-j for j in range(len(sentences))]]

        doc_1 = sentence_separator.join([' '.join(x) for x in sentences])
        
        if include_facts:
            fact_1 = linearize_facts(p_facts)     
        
        for idx, perm_order in enumerate(permutation_ordering):
            doc_2 = sentence_separator.join([' '.join(sentences[i]) for i in perm_order])
            if include_facts:
                fact_2 = linearize_facts([p_facts[i] for i in perm_order])
            
            #create balanced dataset
            json_data = {
                'doc_a': doc_1,
                'doc_b': doc_2,
                'label': -1,
            }

            if include_facts:
                json_data.update({
                    'doc_a_facts': fact_1,
                    'doc_b_facts': fact_2,
                    })

            if idx%2==0:
                json_data = {
                    'doc_a': doc_2,
                    'doc_b': doc_1,
                    'label': 1,
                }

                if include_facts:
                    json_data.update({
                        'doc_a_facts': fact_2,
                        'doc_b_facts': fact_1,
                    })
            res.append(json_data)
    logger.info('single sentence doc count: %d, total doc count: %d, obtained permutation dataset count: %d' % (one_sent_doc_count, total_count, len(res)))
    return res


def process_facts(facts):
    def is_valid_triple(fact_triples):
        status = True
        for y in fact_triples:
            status &= (y.strip()!='' and len(y.strip())!=0)
        return status

    processed_facts = []
    for x in facts:
        if len(x)==3 and is_valid_triple(x):
            processed_facts.append(x)
    return processed_facts 
    
def process_sentences_and_facts(sents, facts, max_sequence_length):
    processed_sents = []
    processed_facts = []
    word_length = 0
    is_truncated = False

    assert len(sents) == len(facts), "mismatch in number of sentences: %d and facts: %d." % (len(sents), len(facts))
    doc_fact_count = 0

    for x, x_facts in zip(sents, facts):
        words = x
        if max_sequence_length > 0 and word_length + len(words) > max_sequence_length:
            is_truncated = True
            break
        word_length += len(words)
        processed_sents.append(x)
        p_facts = process_facts(x_facts)
        doc_fact_count+=len(p_facts)
        processed_facts.append(p_facts)

    return processed_sents, processed_facts, is_truncated, doc_fact_count==0

def featurize_data(tokenizer, fact_tokenizer, include_facts, max_fact_count, max_seq_len, max_fact_seq_len, dataset, logger, exact_count=False):
    max_fact_count_per_doc = 0
    
    position_data = []
    concatenate_all_sentences = []
    concatenate_all_facts = []
    logger.info('post-processing the dataset')
    for doc in tqdm(dataset):
        temp_data = {}

        #label information
        temp_data['label'] = doc['label']

        #concatenate sentences
        temp_data['doc_a_offset'] = len(concatenate_all_sentences)
        concatenate_all_sentences.append(doc['doc_a'])
        if 'doc_b' in doc:
            temp_data['doc_b_offset'] = len(concatenate_all_sentences)
            concatenate_all_sentences.append(doc['doc_b'])
        
        #concatenate facts
        if include_facts:
            doc_a_facts = doc['doc_a_facts']
            if max_fact_count > 0:
                doc_a_facts = doc_a_facts[:max_fact_count]
            temp_data['doc_a_fact_offset'] = len(concatenate_all_facts)
            temp_data['doc_a_fact_count'] = len(doc_a_facts)
            concatenate_all_facts.extend([fact_tokenizer.sep_token.join(x) for x in doc_a_facts])

            doc_b_facts = []
            if 'doc_b_facts' in doc:
                doc_b_facts = doc['doc_b_facts']
                if max_fact_count > 0:
                    doc_b_facts = doc_b_facts[:max_fact_count]
                temp_data['doc_b_fact_offset'] = len(concatenate_all_facts)
                temp_data['doc_b_fact_count'] = len(doc_b_facts)
                concatenate_all_facts.extend([fact_tokenizer.sep_token.join(x) for x in doc_b_facts])
        
            max_fact_count_per_doc = max(max_fact_count_per_doc, len(doc_a_facts), len(doc_b_facts))
        
        position_data.append(temp_data)
    
    if exact_count:
        max_fact_count_per_doc = max_fact_count
    
    logger.info('data preprocessed: %d, sentences preprocessed: %d' % (len(position_data), len(concatenate_all_sentences)))
    
    if include_facts:
        logger.info('fact preprocessed: %d, max_fact_count_per_doc: %d' % (len(concatenate_all_facts), max_fact_count_per_doc))

    sentences_input_ids, sentences_attention_ids = encode_batch(tokenizer, concatenate_all_sentences, max_seq_len)
    if include_facts:
        facts_input_ids, facts_attention_ids = encode_batch(fact_tokenizer, concatenate_all_facts, max_fact_seq_len)
    #as all the facts are padded to specific length
    if include_facts:
        max_fact_seq_len = len(facts_input_ids[0])
    
    max_sent_seq_len = len(sentences_input_ids[0])

    res_data = []
    logger.info('featurizing the datasets..')
    
    for pos in tqdm(position_data):
        temp = {}
        #label information
        temp['label'] = pos['label']
        #process the sentences
        temp['doc_a'] = sentences_input_ids[pos['doc_a_offset']]
        temp['doc_a_mask'] = sentences_attention_ids[pos['doc_a_offset']]
        
        if 'doc_b_offset' in pos and 'doc_b_offset' in pos:
            temp['doc_b'] = sentences_input_ids[pos['doc_b_offset']]
            temp['doc_b_mask'] = sentences_attention_ids[pos['doc_b_offset']]
        
        #process the facts
        if include_facts:
            start, fact_count = pos['doc_a_fact_offset'], pos['doc_a_fact_count']
            temp['doc_a_facts'] = np.zeros((max_fact_count_per_doc, max_fact_seq_len)).tolist()
            temp['doc_a_facts'][:fact_count] = facts_input_ids[start:start+fact_count]
            temp['doc_a_facts_mask'] = np.zeros((max_fact_count_per_doc, max_fact_seq_len)).tolist()
            temp['doc_a_facts_mask'][:fact_count] = facts_attention_ids[start:start+fact_count]
            temp['doc_a_facts_count'] = fact_count

            if 'doc_b_fact_offset' in pos and 'doc_b_fact_count' in pos:
                start, fact_count = pos['doc_b_fact_offset'], pos['doc_b_fact_count']
                temp['doc_b_facts'] = np.zeros((max_fact_count_per_doc, max_fact_seq_len)).tolist()
                temp['doc_b_facts'][:fact_count] = facts_input_ids[start:start+fact_count]
                temp['doc_b_facts_mask'] = np.zeros((max_fact_count_per_doc, max_fact_seq_len)).tolist()
                temp['doc_b_facts_mask'][:fact_count] = facts_attention_ids[start:start+fact_count]
                temp['doc_b_facts_count'] = fact_count
        
        res_data.append(temp)
    
    logger.debug('%d data instance processed. max sent_seq_length: %d' % (len(position_data), max_sent_seq_len))
    if include_facts:
        logger.debug('max fact_seq_length: %d' % (max_fact_seq_len))
    
    return res_data

def featurize_dataset(tokenizer, fact_tokenizer, include_facts, max_fact_count, max_seq_len, max_fact_seq_len, savefile, dataset, logger):
    res_data = featurize_data(tokenizer, fact_tokenizer, include_facts, max_fact_count, max_seq_len, max_fact_seq_len, dataset, logger)
    store_results(savefile, res_data)