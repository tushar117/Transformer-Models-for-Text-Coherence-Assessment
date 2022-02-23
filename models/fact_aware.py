import torch.nn as nn
import torch
from transformers import AutoModel, AutoConfig


class FactAwareModel(nn.Module):
    def __init__(self, args, freeze_emb_layer=True):
        super(FactAwareModel, self).__init__()
        self.tf1 = AutoModel.from_pretrained(args.model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False)
        # use roberta-base only for tf2 
        if args.use_pretrained_tf2:
            # using pretrained transformers for HT 
            self.tf2 = AutoModel.from_pretrained(args.tf2_model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False) 
        else:
            # using non-pretrained transformer layers
            self.tf2 = AutoModel.from_config(AutoConfig.from_pretrained(args.tf2_model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False))

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
        max_facts_count = facts_ids.shape[1]
        # get doc embeddings
        sents_outputs = self.tf1(input_ids=input_ids, attention_mask=attention_mask)[0]
        
        max_sent_fact_seq = facts_count.max().item()+1
        sent_fact_embeddings = torch.zeros(batch_size, max_sent_fact_seq, self.tf1.config.hidden_size).to(input_ids.device)
        sent_fact_attention_mask = torch.zeros(batch_size, max_sent_fact_seq, dtype=torch.long).to(input_ids.device)

        for batch_idx in range(batch_size):
            #combine the sentences and facts at the lower transformer
            #retrieving the [CLS] token for sentence
            sent_fact_embeddings[batch_idx, 0] = sents_outputs[batch_idx][0]
            local_facts_count = facts_count[batch_idx].item()
            facts_outputs = self.tf2(input_ids=facts_ids[batch_idx][:local_facts_count], attention_mask=facts_mask[batch_idx][:local_facts_count])[0]
            #retrieving the [CLS] token for fact extracted
            sent_fact_embeddings[batch_idx, 1:local_facts_count+1] = facts_outputs[:, 0, :]
            sent_fact_attention_mask[batch_idx, :local_facts_count+1] = 1

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