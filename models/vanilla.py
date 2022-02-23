import torch.nn as nn
from transformers import AutoModel, AutoConfig 


class TransformerModel(nn.Module):
    def __init__(self, args, freeze_emb_layer=True):
        super(TransformerModel, self).__init__()
        # using tf2 as variable name to maintain uniformity across the code
        self.tf2 = AutoModel.from_pretrained(args.model_name, hidden_dropout_prob=args.dropout_rate, add_pooling_layer=False)
        self.args = args
        if args.freeze_emb_layer and freeze_emb_layer:
            self.layer_freezing()

    def forward(self, input_data):
        input_ids, attention_mask = input_data
        # get last layer sequence output
        output1 = self.tf2(input_ids=input_ids, attention_mask=attention_mask)[0]
        #return the vector represenation for [CLS] token from last layer output
        return output1[:, 0, :]
    
    def layer_freezing(self, freeze_layers=[], freeze_embedding=True):
        if freeze_embedding:
            for param in list(self.tf2.embeddings.parameters()):
                param.requires_grad = False
            self.args.logger.info("frozed embedding layer")
        
        for layer_idx in freeze_layers:
            for param in list(self.tf2.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False
            self.args.logger.info("frozed internal layer: %d" % layer_idx) 