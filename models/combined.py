import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig 
import torch
import numpy as np


class CombinedModel(nn.Module):
    def __init__(self, args, freeze_emb_layer=True):
        super(CombinedModel, self).__init__()
        self.tf1 = AutoModel.from_pretrained(args.model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False)
        if args.use_pretrained_tf2:
            #using pretrained transformers for HT 
            self.tf2 = AutoModel.from_pretrained(args.tf2_model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False) 
        else:
            #also try with vanilla transformer layers
            self.tf2 = AutoModel.from_config(AutoConfig.from_pretrained(args.tf2_model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False)) 
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.sentence_pooling = args.sentence_pooling
        self.end_token_id = tokenizer.sep_token_id
        if self.sentence_pooling == 'attention':
            self.linear_weight = nn.Linear(self.tf1.config.hidden_size, self.tf1.config.hidden_size)
            self.linear_value = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(self.tf1.config.hidden_size, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.args = args
        if args.freeze_emb_layer and freeze_emb_layer:
            self.layer_freezing()

    def _get_sentences_offsets(self, sentence_sep_tokens_pos):
        sentence_offsets = []
        start_index = 1
        sorted_indexes, _ = torch.nonzero(sentence_sep_tokens_pos).squeeze(1).sort()
        for index in sorted_indexes:
            end_index = index.item() - 1
            sentence_offsets.append([start_index, end_index])
            start_index = index.item() + 1
        return sentence_offsets

    def forward(self, input_data):
        input_ids, attention_mask, facts_ids, facts_mask, facts_count = input_data
        batch_size = input_ids.shape[0]
        end_token_lookup = input_ids==self.end_token_id
        # get last layer sequence output
        output1 = self.tf1(input_ids=input_ids, attention_mask=attention_mask)[0]
        prior_max_sent_count = end_token_lookup.sum(axis=-1).max()
        # cover cases for sentence related embedding 
        max_sent_count = max(prior_max_sent_count.item(), 1)
        # cover cases for the facts related embedding
        max_sent_fact_seq = facts_count.max().item() + max_sent_count
        sent_fact_embeddings = torch.zeros(batch_size, max_sent_fact_seq, self.tf1.config.hidden_size).to(input_ids.device)
        sent_fact_attention_mask = torch.zeros(batch_size, max_sent_fact_seq, dtype=torch.long).to(input_ids.device)

        for batch_idx in range(batch_size):
            # processing sentence information
            local_sent_count = end_token_lookup[batch_idx].sum(axis=-1).item()
            if local_sent_count == 0:
                #if sentence separator is not present in the document use [CLS] token to mark whole document as one sentence
                local_sent_count = 1
                sent_fact_embeddings[batch_idx, :local_sent_count] = output1[batch_idx][0]
            else:
                #define different type of sub-word pooling for sentence embedding
                if self.sentence_pooling is not 'none':
                    with torch.no_grad():
                        #get individual sentence offset
                        sentences_offsets = self._get_sentences_offsets(end_token_lookup[batch_idx])
                    
                    for sent_indx in range(local_sent_count):
                        start_index, end_index = sentences_offsets[sent_indx]
                        if self.sentence_pooling == 'sum':
                            sent_fact_embeddings[batch_idx, sent_indx] = torch.sum(output1[batch_idx][start_index:end_index+1], axis=0)
                        elif self.sentence_pooling == 'mean':
                            sent_fact_embeddings[batch_idx, sent_indx] = torch.mean(output1[batch_idx][start_index:end_index+1], axis=0)
                        elif self.sentence_pooling == 'max':
                            sent_fact_embeddings[batch_idx, sent_indx], _ = torch.max(output1[batch_idx][start_index:end_index+1], axis=0)
                        elif self.sentence_pooling == 'min':
                            sent_fact_embeddings[batch_idx, sent_indx], _ = torch.min(output1[batch_idx][start_index:end_index+1], axis=0)
                        elif self.sentence_pooling == 'attention':
                            #transform the sub-word representation
                            subword_rep = torch.tanh(self.linear_weight(output1[batch_idx][start_index:end_index+1]))
                            attention_weights = torch.softmax(subword_rep.mm(self.linear_value), dim=0)
                            #calculate the weighted representation using attention weights
                            sent_fact_embeddings[batch_idx, sent_indx] = torch.sum((attention_weights * output1[batch_idx][start_index:end_index+1]), dim=0)
                else:
                    sent_fact_embeddings[batch_idx, :local_sent_count] = output1[batch_idx][end_token_lookup[batch_idx]]
            
            # processing factual information
            local_facts_count = facts_count[batch_idx].item()
            facts_outputs = self.tf2(input_ids=facts_ids[batch_idx][:local_facts_count], attention_mask=facts_mask[batch_idx][:local_facts_count])[0]
            #retrieving the [CLS] token for fact extracted
            sent_fact_embeddings[batch_idx, local_sent_count:local_sent_count+local_facts_count] = facts_outputs[:, 0, :]
    
            #sentence+fact masking for next layer
            sent_fact_attention_mask[batch_idx, :local_sent_count+local_facts_count] = 1

        #get last layer sequence output
        output2 = self.tf2(inputs_embeds=sent_fact_embeddings, attention_mask=sent_fact_attention_mask)[0]
        #return the vector represenation for [CLS] token from last layer output
        return output2[:, 0, :]
    
    def layer_freezing(self, freeze_layers=[], freeze_embedding=True):
        if freeze_embedding:
            for param in list(self.tf1.embeddings.parameters()):
                param.requires_grad = False
            self.args.logger.info("frozed embedding layer")
        
        for layer_idx in freeze_layers:
            for param in list(self.tf1.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False
            self.args.logger.info("frozed internal layer: %d" % layer_idx)