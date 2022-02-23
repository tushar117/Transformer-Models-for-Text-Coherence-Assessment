import os, sys
import json
import torch.nn as nn
import torch
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import WandbLogger
import random
import numpy as np
from featurize_dataset import featurize_train_dataset, featurize_test_dataset
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, cohen_kappa_score
from utility import calculate_actual_score, load_txt
import transformers
from dataloader import get_dataset_loaders
import torch.nn.functional as F

# required to access the python modules present in project directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# now we can import the all modules present in project folder
from logger import MyLogger, LOG_LEVELS
from models import *
from utils.common import load_file, full_clean_directory
from dataset_processor.featurize_dataset import MTLFeaturizer, CombinedFeaturizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prompt_mean = {
    0:0.14950598531856235,
    1:0.1538133649558774,
    2:0.1548593946093211,
    3:0.2716402537607249,
    4:0.31317225562561973,
    5:0.2426380130829415,
    6:0.24259018941416405,
    7:0.15279627847213903,
    8:0.09582535490848867,
}

class PairWiseSentenceRanking(nn.Module):
    """Head for pairwise sentence ranking task"""
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(PairWiseSentenceRanking, self).__init__()
        #document transformation
        self.phi = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, sent_a, sent_b):
        return self.phi(self.dropout(sent_a)), self.phi(self.dropout(sent_b))

class TexClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, hidden_size, num_labels, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = features  # takes [CLS] token representation as input
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def get_task_specific_dataset_index(dataset_ids):
        entailment_data_idx = []
        task_specific_data_idx = []
        for idx, d_id in enumerate(dataset_ids):
            # fetching the indexes for different tasks
            if d_id == 0:
                entailment_data_idx.append(idx)
            else:
                task_specific_data_idx.append(idx)
        return entailment_data_idx, task_specific_data_idx

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EssayRankingModel(nn.Module):
    def __init__(self, args):
        super(EssayRankingModel, self).__init__()
        self.config_args = args
        # load transformer architecture
        self.doc_encoder = get_module(args.arch, base_arch=args.mtl_base_arch)(args)
        self.task_head = PairWiseSentenceRanking(self.doc_encoder.tf2.config.hidden_size, args.dropout_rate)
        # handling the multi-task learning approach
        if self.config_args.arch in ['mtl', 'combined']:
            # adding textual entailment task head
            self.te_task_head = TexClassificationHead(self.doc_encoder.tf2.config.hidden_size, 2, args.dropout_rate)

    def forward(self, input_data):
        output_logits = []
        # dummy indicator (when MTL is inactive) variable for task specific data instances
        task_specific_data_idx = [i for i in range(input_data[0].shape[0])]

        if self.config_args.arch in ['mtl', 'combined'] and not args.inference:
            # gather textual entailment dataset i.e. d_id == 0
            d_ids = input_data[0]             
            entailment_data_idx, task_specific_data_idx = get_task_specific_dataset_index(d_ids)
            if len(entailment_data_idx):
                entailment_data = [z for z in map(lambda x: x[entailment_data_idx], input_data[1:])]
                # caution : need to change below code "entailment_data[:2]" according to the transformers model used for Multi-task-learning
                # presently it's set to work with vanilla transformer model when 'mtl' is active.
                if self.config_args.arch=='mtl' and self.config_args.mtl_base_arch in ['vanilla', 'hierarchical']:
                    entailment_doc_rep = self.doc_encoder(entailment_data[:2])
                else:
                    entailment_doc_rep = self.doc_encoder(entailment_data[:5])
                output_logits.append(self.te_task_head(entailment_doc_rep))    
            # assigning the task_specific_data to input_data variable for normal flow
            input_data = [z for z in map(lambda x: x[task_specific_data_idx], input_data[1:])]
        
        if len(task_specific_data_idx):
            assert len(input_data)%2 == 0
            mid = int(len(input_data)/2)
            doc_a_data = input_data[:mid]
            doc_b_data = input_data[mid:]
            coherence_a = self.doc_encoder(doc_a_data)
            coherence_b = self.doc_encoder(doc_b_data)
            output_logits.append(self.task_head(coherence_a, coherence_b))
        return output_logits

class EssayScorer(nn.Module):
    "Head for sentence score task (regression)"
    def __init__(self, hidden_size, hidden_dropout_prob, bias=None):
        super().__init__()
        self.bias = bias
        if bias is None:
            self.dense = nn.Linear(hidden_size, 1)
        else:
            self.dense = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, features):
        x = features  # takes [CLS] token representation as input
        x = self.dropout(x)
        x = self.dense(x)
        if self.bias is not None:
            x = x + self.bias
        return torch.sigmoid(x)

class EssayScoringModel(nn.Module):
    def __init__(self, args):
        super(EssayScoringModel, self).__init__()
        self.config_args = args
        # use vanilla transformer for the essay scoring
        self.essay_encoder = get_module('vanilla')(args, freeze_emb_layer=True)
        hidden_size = self.essay_encoder.tf2.config.hidden_size
        if self.config_args.enable_coherence_signal > 0:
            # considering the coherence vectors are of 768 dimension
            hidden_size += 768
        self.task_head = EssayScorer(hidden_size, args.dropout_rate, bias=prompt_mean[args.prompt_id])
    
    def forward(self, input_data):
        if self.config_args.enable_coherence_signal > 0:
            coherence_rep = input_data[-1].requires_grad_(False)
            essay_rep = torch.cat([self.essay_encoder(input_data[:-1]), coherence_rep], dim=-1)
        else:
            essay_rep = self.essay_encoder(input_data)
        return self.task_head(essay_rep)

def store_text_coherence_vectors(args, fold_id, model, iterator):
    # set the model in eval mode so the dropout and other training parameter will not be effective
    model.eval()

    global_essay_id = []
    global_doc_rep = []

    for batch in iterator:
        online_logger_data = {}
        essay_id, input_data = batch[0].to(device), [x.to(device) for x in batch[1:]] 
        doc_rep = model.doc_encoder(input_data)
        global_essay_id.append(essay_id)
        global_doc_rep.append(doc_rep)   
        
    global_essay_id = torch.cat(global_essay_id).tolist()
    global_doc_rep = torch.cat(global_doc_rep).tolist()

    assert len(global_essay_id) == len(global_doc_rep), "mismatch between essay id and doc representation"
    final_res = {}
    for eid, erep in zip(global_essay_id, global_doc_rep):
        final_res[eid] = erep
    
    save_filename = os.path.join(args.processed_dataset_path, "%s-coherence-vector-%d.json"%(args.arch, fold_id))
    with open(save_filename, 'w') as dfile:
        json.dump(final_res, dfile)
    args.logger.info('successfully stored the coherence vector to location : %s' % save_filename)
    # upload coherence vector file to wandb
    wandb.save(save_filename)

def evaluate_sentence_ordering_model(args, fold_id, model, iterator, set_type='val', inference=False):
    # set the model in eval mode so the dropout and other training parameter will not be effective
    model.eval()
    batch_loss = []
    batch_task_loss = []

    # essay scoring
    predicted_labels = []
    actual_labels = []
    
    # textual entailment prediction
    if args.arch in ['mtl', 'combined'] and not inference:
        te_prediction = []
        te_actual = []
        batch_te_loss = []

    for batch in iterator:
        online_logger_data = {}
        input_data, label_ids = [x.to(device) for x in batch[:-1]], batch[-1].to(device)
        # dummy indicator (when MTL is inactive) variable for task specific data instances
        task_specific_data_idx = [i for i in range(label_ids.view(-1).size(0))]
        model_outputs = model(input_data)
        loss = torch.tensor(0.0).to(device)
        task_loss = torch.tensor(0.0).to(device)

        # architecture specific (MTL) loss calculation
        if args.arch in ["mtl", "combined"] and not inference:
            d_ids = input_data[0]
            
            with torch.no_grad():
                entailment_data_idx, task_specific_data_idx = get_task_specific_dataset_index(d_ids)
            
            if len(entailment_data_idx):
                te_pred = model_outputs[0]
                entailment_label_ids = label_ids[entailment_data_idx].long()
                te_loss = F.cross_entropy(te_pred, entailment_label_ids)
                with torch.no_grad():
                    # updating the textual entailment loss to calculate the dataset wide loss (epoch loss)
                    batch_te_loss.append(te_loss.item())
                    te_acc = accuracy_score(entailment_label_ids.view(-1).tolist(), te_pred.softmax(dim=-1).view(-1, 2).max(dim=-1)[1].tolist())
                    # for calculating the dataset wide accuracy (i.e per epcoh)
                    te_prediction.append(te_pred.softmax(dim=-1).view(-1, 2))
                    te_actual.append(entailment_label_ids.view(-1))
                # updating logs into online logger
                online_logger_data.update({'batch_%s_te_loss_%d'%(set_type, fold_id): te_loss.item(), 'batch_%s_te_acc_%d'%(set_type, fold_id): te_acc})
                loss += te_loss
            online_logger_data.update({"te_%s_data_count_%d"%(set_type, fold_id): len(entailment_data_idx),
                                    "ranking_%s_data_count_%d"%(set_type, fold_id): len(task_specific_data_idx)})
            # assigning the task_specific_labels to label_ids variable for normal flow
            label_ids = label_ids[task_specific_data_idx]

        if len(task_specific_data_idx):
            # task specific loss calculation
            model_output = model_outputs[-1]
            # updating the task specific loss to calculate the dataset wide loss (epoch loss)
            with torch.no_grad():
                # sentence score for 'sentence a' and 'sentence b' are 'phi_a' and 'phi_b' respectively
                phi_a, phi_b = model_output
                task_loss = F.margin_ranking_loss(phi_a, phi_b, label_ids, margin=args.margin)
                pred_res = -1*torch.ones_like(label_ids)
                pred_res[phi_a.detach().requires_grad_(False).view(-1) > phi_b.detach().requires_grad_(False).view(-1)] = 1

                batch_task_loss.append(task_loss.item())
                predicted_labels.append(pred_res)
                actual_labels.append(label_ids)
        
        loss += task_loss
        
        with torch.no_grad():
            batch_loss.append(loss.item())
            if args.arch in ["mtl", "combined"] and not inference:
                online_logger_data.update({'batch_%s_ranking_loss_%d'%(set_type, fold_id): task_loss})
            
            online_logger_data.update({'batch_%s_combined_loss_%d'%(set_type, fold_id): loss})
            wandb.log(online_logger_data)

    # calculating the dataset wide loss and evaluation metric
    all_pred_label = torch.cat(predicted_labels).squeeze(-1).tolist()
    all_actual_label = torch.cat(actual_labels).tolist()
    avg_loss = np.mean(batch_loss)

    # caculating weighted kappa score
    pra = accuracy_score(all_actual_label, all_pred_label)
    args.logger.info(" %s_loss: %f, %s_accuracy: %f" % (set_type, avg_loss, set_type, pra))

    epoch_end_logger_data = {
        'avg_%s_combined_loss_%d'%(set_type, fold_id): avg_loss,
        'PRA_%s_%d'%(set_type, fold_id): pra,
    }

    if (args.arch == "mtl" or ( args.arch == "combined" and args.disable_mtl<=0 )) and not inference:
        avg_task_loss = np.mean(batch_task_loss)
        avg_te_loss = np.mean(batch_te_loss)
        all_te_pred = torch.cat(te_prediction).max(dim=-1)[1].tolist()
        all_te_actual_label = torch.cat(te_actual).tolist()
        te_accuracy = accuracy_score(all_te_actual_label, all_te_pred)
        args.logger.info(" %s_ranking_loss: %f, %s_te_loss:%f, %s_te_accuracy: %f" % (set_type, avg_task_loss, set_type, avg_te_loss, set_type, te_accuracy))
        epoch_end_logger_data.update({
            'avg_%s_te_loss_%d'%(set_type, fold_id): avg_te_loss,
            '%s_te_accuracy'%set_type: te_accuracy,
        })
    
    wandb.log(epoch_end_logger_data)
    return avg_loss, pra

def sentence_ordering_training_loop(args, fold_id, model, train_iterator, valid_iterator):
    # set the model in train mode so the dropout and other training parameter will be effective
    model.train()
    # confuguring the optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.rank_learning_rate)
    
    args.logger.info('executing the sentence ordering training loop..')
    max_val_pra = 0.0

    for epoch in range(1, args.rank_epochs+1):
        args.logger.info('=='*30)
        args.logger.info('[Train sentence ordering] Epoch [fold - %d / %d] : %d / %d' % (fold_id, args.kfolds, epoch, args.rank_epochs))
        batch_loss = []
        batch_task_loss = []

        # essay scoring
        predicted_labels = []
        actual_labels = []
        # textual entailment prediction
        if args.arch in ['mtl', 'combined']:
            te_prediction = []
            te_actual = []
            batch_te_loss = []

        pbar = tqdm(train_iterator)
        for batch in pbar:
            online_logger_data = {}
            input_data, label_ids = [x.to(device) for x in batch[:-1]], batch[-1].to(device)
            # dummy indicator (when MTL is inactive) variable for task specific data instances
            task_specific_data_idx = [i for i in range(label_ids.view(-1).size(0))]
            entailment_data_idx = []
            model_outputs = model(input_data)
            loss = torch.tensor(0.0).to(device)
            task_loss = torch.tensor(0.0).to(device)

            optimizer.zero_grad()
            # architecture specific (MTL) loss calculation
            if args.arch in ["mtl", "combined"]:
                d_ids = input_data[0]
                
                with torch.no_grad():
                    entailment_data_idx, task_specific_data_idx = get_task_specific_dataset_index(d_ids)
                
                if len(entailment_data_idx):
                    te_pred = model_outputs[0]
                    entailment_label_ids = label_ids[entailment_data_idx].long()
                    te_loss = F.cross_entropy(te_pred, entailment_label_ids)
                    with torch.no_grad():
                        # updating the textual entailment loss to calculate the dataset wide loss (epoch loss)
                        batch_te_loss.append(te_loss.item())
                        te_acc = accuracy_score(entailment_label_ids.view(-1).tolist(), te_pred.softmax(dim=-1).view(-1, 2).max(dim=-1)[1].tolist())
                        # for calculating the dataset wide accuracy (i.e per epcoh)
                        te_prediction.append(te_pred.softmax(dim=-1).view(-1, 2))
                        te_actual.append(entailment_label_ids.view(-1))
                    # updating logs into online logger
                    online_logger_data.update({'batch_train_te_loss_%d'%fold_id: te_loss.item(), 'batch_train_te_acc_%d'%fold_id: te_acc})
                    loss += te_loss
                online_logger_data.update({"te_data_count_%d"%fold_id: len(entailment_data_idx),
                                        "ranking_data_count_%d"%fold_id: len(task_specific_data_idx)})
                # assigning the task_specific_labels to label_ids variable for normal flow
                label_ids = label_ids[task_specific_data_idx]

            if len(task_specific_data_idx):
                # task specific loss calculation
                model_output = model_outputs[-1]
                # sentence score for 'sentence a' and 'sentence b' are 'phi_a' and 'phi_b' respectively
                phi_a, phi_b = model_output
                task_loss = F.margin_ranking_loss(phi_a, phi_b, label_ids, margin=args.margin)
                with torch.no_grad():
                    pred_res = -1*torch.ones_like(label_ids)
                    pred_res[phi_a.detach().requires_grad_(False).view(-1) > phi_b.detach().requires_grad_(False).view(-1)] = 1

                    batch_task_loss.append(task_loss.item())
                    predicted_labels.append(pred_res)
                    actual_labels.append(label_ids)
            
            loss += task_loss
            # updating the global loss
            loss.backward()
            if args.clip_norm>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            
            with torch.no_grad():
                batch_loss.append(loss.item())
                pbar.set_description("loss: %f" % (loss.item()))
                if args.arch in ["mtl", "combined"]:
                    online_logger_data.update({'batch_train_ranking_loss_%d'%fold_id: task_loss})
                
                online_logger_data.update({'batch_train_combined_loss_%d'%fold_id: loss})
                wandb.log(online_logger_data)

        # calculating the dataset wide loss and evaluation metric
        all_pred_label = torch.cat(predicted_labels).squeeze(-1).tolist()
        all_actual_label = torch.cat(actual_labels).tolist()
        avg_loss = np.mean(batch_loss)

        # caculating pairwise ranking accuracy
        pra = accuracy_score(all_actual_label, all_pred_label)
        args.logger.info(" train_combined_loss: %f, accuracy: %f" % (avg_loss, pra))
        
        epoch_end_logger_data = {
            'avg_train_combined_loss_%d'%fold_id: avg_loss,
            'PRA_train_%d'%fold_id: pra,
        }

        if args.arch == "mtl" or ( args.arch == "combined" and args.disable_mtl<=0 ):
            all_te_pred = torch.cat(te_prediction).max(dim=-1)[1].tolist()
            all_te_actual_label = torch.cat(te_actual).tolist()       
            avg_task_loss = np.mean(batch_task_loss)
            avg_te_loss = np.mean(batch_te_loss)
            te_accuracy = accuracy_score(all_te_actual_label, all_te_pred)
            args.logger.info(" train_ranking_loss: %f, train_te_loss:%f, train_te_accuracy: %f" % (avg_task_loss, avg_te_loss, te_accuracy))
            epoch_end_logger_data.update({
                'avg_train_te_loss_%d'%fold_id: avg_te_loss,
                'train_te_accuracy': te_accuracy,
            })
        
        wandb.log(epoch_end_logger_data)

        # model evaluation
        with torch.no_grad():
            _, val_pra = evaluate_sentence_ordering_model(args, fold_id, model, valid_iterator)
            if max_val_pra <= val_pra:
                args.logger.info('[validation set] new best val accuracy: %f, previous best: %f' % (val_pra, max_val_pra))
                full_clean_directory(args.model_save_path)
                with open(os.path.join(args.model_save_path, '%s-%s-ranking-epoch-%d.pt' % (args.arch, args.model_name.replace('/', '-'), epoch)), 'wb') as model_cache_path:
                    torch.save(model.state_dict(), model_cache_path)
                max_val_pra = val_pra

def evaluate_model(args, fold_id, model, iterator, set_type='val', inference=False):
    # set the model in eval mode so the dropout and other training parameter will not be effective
    model.eval()
    batch_loss = []
    batch_task_loss = []

    # essay scoring
    predicted_score = []
    actual_normalized_score = []
    global_prompt_ids = []

    for batch in iterator:
        online_logger_data = {}
        prompt_ids, input_data, label_ids = batch[0].to(device), [x.to(device) for x in batch[1:-1]], batch[-1].to(device)
        loss = torch.tensor(0.0).to(device)
        task_loss = torch.tensor(0.0).to(device)

        model_output = model(input_data)
        if args.enable_huber_loss:
            task_loss = F.smooth_l1_loss(model_output.view(-1), label_ids.view(-1))
        else:
            task_loss = F.mse_loss(model_output.view(-1), label_ids.view(-1))
        # updating the task specific loss to calculate the dataset wide loss (epoch loss)
        with torch.no_grad():
            batch_task_loss.append(task_loss.item())
            predicted_score.append(model_output)
            actual_normalized_score.append(label_ids)
            global_prompt_ids.append(prompt_ids)
        
        loss += task_loss
        
        with torch.no_grad():
            batch_loss.append(loss.item())
            if args.arch in ["mtl", "combined"] and not inference:
                online_logger_data.update({'batch_%s_scoring_loss_%d'%(set_type, fold_id): task_loss})
            
            online_logger_data.update({'batch_%s_scoring_loss_%d'%(set_type, fold_id): loss})
            wandb.log(online_logger_data)

    # calculating the dataset wide loss and evaluation metric
    all_pred_score = torch.cat(predicted_score).squeeze(-1).tolist()
    all_norm_score = torch.cat(actual_normalized_score).tolist()
    all_prompt_id = torch.cat(global_prompt_ids).tolist()
    avg_loss = np.mean(batch_loss)

    gold_pred_score = calculate_actual_score(all_pred_score, all_prompt_id)
    gold_actual_score = calculate_actual_score(all_norm_score, all_prompt_id)
    # caculating weighted kappa score
    wks = cohen_kappa_score(gold_actual_score, gold_pred_score, weights='quadratic')
    args.logger.info(" %s_scoring_loss: %f, %s_weighted kappa score: %f" % (set_type, avg_loss, set_type, wks))

    epoch_end_logger_data = {
        'avg_%s_scoring_loss_%d'%(set_type, fold_id): avg_loss,
        'wks_%s_%d'%(set_type, fold_id): wks,
    }
    
    wandb.log(epoch_end_logger_data)
    return avg_loss, wks

def training_loop(args, fold_id, model, train_iterator, valid_iterator):
    # set the model in train mode so the dropout and other training parameter will be effective
    model.train()
    # confuguring the optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.score_weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.score_learning_rate)
    
    args.logger.info('executing the essay scoring training loop..')
    max_kappa_score= -1

    for epoch in range(1, args.score_epochs+1):
        args.logger.info('=='*30)
        args.logger.info('[Train essay scoring] Epoch [fold - %d / %d] : %d / %d' % (fold_id, args.kfolds, epoch, args.score_epochs))
        batch_loss = []
        batch_task_loss = []

        # essay scoring
        predicted_score = []
        actual_normalized_score = []
        global_prompt_ids = []
        # textual entailment prediction
        
        pbar = tqdm(train_iterator)
        for batch in pbar:
            online_logger_data = {}
            prompt_ids, input_data, label_ids = batch[0].to(device), [x.to(device) for x in batch[1:-1]], batch[-1].to(device)
            loss = torch.tensor(0.0).to(device)
            task_loss = torch.tensor(0.0).to(device)

            optimizer.zero_grad()

            # task specific loss calculation
            model_output = model(input_data)
            if args.enable_huber_loss:
                task_loss = F.smooth_l1_loss(model_output.view(-1), label_ids.view(-1))
            else:
                task_loss = F.mse_loss(model_output.view(-1), label_ids.view(-1))
            # updating the task specific loss to calculate the dataset wide loss (epoch loss)
            with torch.no_grad():
                batch_task_loss.append(task_loss.item())
                predicted_score.append(model_output)
                actual_normalized_score.append(label_ids)
                global_prompt_ids.append(prompt_ids)
            
            loss += task_loss
            # updating the global loss
            loss.backward()
            if args.score_clip_norm>0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.score_clip_norm)
            optimizer.step()
            
            with torch.no_grad():
                batch_loss.append(loss.item())
                pbar.set_description("loss: %f" % (loss.item()))
                online_logger_data.update({'batch_train_scoring_loss_%d'%fold_id: loss})
                wandb.log(online_logger_data)

        # calculating the dataset wide loss and evaluation metric
        all_pred_score = torch.cat(predicted_score).squeeze(-1).tolist()
        all_norm_score = torch.cat(actual_normalized_score).tolist()
        all_prompt_id = torch.cat(global_prompt_ids).tolist()
        avg_loss = np.mean(batch_loss)

        gold_pred_score = calculate_actual_score(all_pred_score, all_prompt_id)
        gold_actual_score = calculate_actual_score(all_norm_score, all_prompt_id)
        # caculating weighted kappa score
        wks = cohen_kappa_score(gold_actual_score, gold_pred_score, weights='quadratic')
        args.logger.info(" train_loss: %f, weighted kappa score: %f" % (avg_loss, wks))
        
        epoch_end_logger_data = {
            'avg_train_scoring_loss_%d'%fold_id: avg_loss,
            'wks_train_%d'%fold_id: wks,
        }
        
        wandb.log(epoch_end_logger_data)

        # model evaluation
        with torch.no_grad():
            _, val_kappa_score = evaluate_model(args, fold_id, model, valid_iterator)
            if max_kappa_score <= val_kappa_score:
                args.logger.info('[validation set] new best weighted kappa score: %f, previous best: %f' % (val_kappa_score, max_kappa_score))
                full_clean_directory(args.model_save_path)
                with open(os.path.join(args.model_save_path, '%s-%s-epoch-%d.pt' % (args.arch, args.model_name.replace('/', '-'), epoch)), 'wb') as model_cache_path:
                    torch.save(model.state_dict(), model_cache_path)
                max_kappa_score = val_kappa_score

def filter_dataset(prompt_id, dataset):
    return dataset if prompt_id==0 else [x for x in dataset if x['essay_set']==prompt_id]

def load_checkpoint(model, args, sentence_ordering=False):
    prefix_str = '%s-%s-epoch-'%(args.arch, args.model_name.replace('/', '-'))
    if sentence_ordering:
        prefix_str = '%s-%s-ranking-epoch-'%(args.arch, args.model_name.replace('/', '-'))

    args.logger.info('loading the %s checkpoint ...' % ('sentence ranking' if sentence_ordering else 'sentence scoring'))
    file_list = [x for x in os.listdir(args.model_save_path) if x.startswith(prefix_str)]
    
    assert len(file_list)>0, "no checkpoint exists"
    if len(file_list)!=1:
        args.logger.critical('found %d files in checkpoint dir %s' % (len(file_list), args.model_save_path))
    # if more than one checkpoint exists, then randomly select one of it
    file_name = os.path.join(args.model_save_path, random.choice(file_list))
    if sentence_ordering:
        # find the matching layer from present model
        doc_encoder = ["doc_encoder.%s"%k for k in model.doc_encoder.state_dict()]
        with open(file_name, 'rb') as wfile:
            pretrained_dict = torch.load(wfile)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in doc_encoder}
        model.load_state_dict(pretrained_dict, strict=False)
    else:
        model.load_state_dict(torch.load(file_name))
    args.logger.info('successfully loaded the checkpoint : %s' % file_name)

def match_data_split(total_data, data_type_index):
    final_dataset = []
    for x in total_data:
        if x['esssay_id'] not in data_type_index:
            continue
        final_dataset.append(x)
    return final_dataset

def init(args):
    test_kappa_score_list = []
    args.logger.critical('>>>>>>>>>>> [DEVICE] : %s <<<<<<<<<<<' % device)
    # loading the dataset
    raw_dataset = load_file(os.path.join(args.processed_dataset_path, 'AES', 'train.jsonl'))
    filtered_dataset = filter_dataset(args.prompt_id, raw_dataset)
    # creation k-fold dataset
    kf = KFold(n_splits=args.kfolds, random_state=42)
    fold_index = 0
  
    random.seed(42)
    # if kfold is 5 then train, dev, test split is 60:20:20
    wrapper_featurizer = None
    if args.arch in ['mtl', 'combined']:
        args.task = "essay_scoring"
        args.corpus = "asap"
        wrapper_featurizer = MTLFeaturizer(args) if args.arch == 'mtl' else CombinedFeaturizer(args)

    # train_dev_index, test_index
    for fold_count in range(args.kfolds):
        args.logger.info('loading the train, val and test indices')
        fold_map = {
            'train': [int(tz) for tz in load_txt(os.path.join(args.processed_dataset_path, 'AES', 'fold_%d'%fold_count, 'train_ids.txt'))],
            'dev': [int(tz) for tz in load_txt(os.path.join(args.processed_dataset_path, 'AES', 'fold_%d'%fold_count, 'dev_ids.txt'))],
            'test': [int(tz) for tz in load_txt(os.path.join(args.processed_dataset_path, 'AES', 'fold_%d'%fold_count, 'test_ids.txt'))]
        }

        fold_index = fold_count + 1
        args.logger.info('>> working on the %d datafold <<' % fold_index)
        # extracting the dev and train index
        dev_index_data = match_data_split(filtered_dataset, fold_map['dev']) 
        train_index_data = match_data_split(filtered_dataset, fold_map['train'])
        test_index_data = match_data_split(filtered_dataset, fold_map['test'])
        args.logger.info('dataset count | train : %d [%.2f], dev : %d [%.2f] and test : %d [%.2f]' %(len(train_index_data), len(train_index_data)/len(filtered_dataset), 
                                                                                            len(dev_index_data), len(dev_index_data)/len(filtered_dataset),
                                                                                            len(test_index_data), len(test_index_data)/len(filtered_dataset)))
        if args.enable_coherence_signal>0 and args.coherence_vector:
            args.logger.info("using the pre-calculated coherence vectors : %s" % os.path.abspath(args.coherence_vector))
        if args.enable_coherence_signal>0 and args.coherence_vector is None:
            # getting sentence ranking dataset
            featurize_train_dataset(args, 'train', train_index_data, fold_index, sentence_ordering=True)
            featurize_train_dataset(args, 'dev', dev_index_data, fold_index, sentence_ordering=True)

            # invoking additional featurizer if MTL or combined artitecture is selected
            if wrapper_featurizer:
                wrapper_featurizer.featurize_dataset(inference=False)

            # creating the train ,validation and test dataloader
            train_iterator = get_dataset_loaders(os.path.join(args.processed_dataset_path, 'featurized_dataset', 'train.jsonl'), batch_size=args.rank_batch_size, float_label=False)
            valid_iterator = get_dataset_loaders(os.path.join(args.processed_dataset_path, 'featurized_dataset', 'dev.jsonl'), batch_size=args.rank_batch_size, float_label=False)
            
            # intializing the essay ranking model
            model = EssayRankingModel(args).to(device)
            args.logger.info("<<<<<<<< intializing the essay ranking model >>>>>>>>>")
            args.logger.info('Ranking model has %d trainable parameters' % (count_parameters(model)))
            args.logger.info(model)

            sentence_ordering_training_loop(args, fold_index, model, train_iterator, valid_iterator)
            
            # store textual coherence vector for the each essay for a selected prompt
            featurize_test_dataset(args, test_index_data, fold_index, sentence_ordering=True)
            test_iterator = get_dataset_loaders(os.path.join(args.processed_dataset_path, 'featurized_dataset', 'test.jsonl'), batch_size=args.rank_batch_size)
            with torch.no_grad():
                # loac the checkpoint with best validation PRA
                load_checkpoint(model, args, sentence_ordering=True)
                store_text_coherence_vectors(args, fold_index, model, test_iterator)
        
        # getting essay scoring dataset
        featurize_train_dataset(args, 'train', train_index_data, fold_index)
        featurize_train_dataset(args, 'dev', dev_index_data, fold_index)
        # loading the essay scoring dataset
        scoring_train_iterator = get_dataset_loaders(os.path.join(args.processed_dataset_path, 'featurized_dataset', 'train.jsonl'), batch_size=args.score_batch_size)
        scoring_valid_iterator = get_dataset_loaders(os.path.join(args.processed_dataset_path, 'featurized_dataset', 'dev.jsonl'), batch_size=args.score_batch_size)
        
        model = EssayScoringModel(args).to(device)
        
        args.logger.info("<<<<<<<< intializing the essay scoring model >>>>>>>>>")
        # load trained sentence ordering
        args.logger.info('scoring model has %d trainable parameters' % (count_parameters(model)))
        args.logger.info(model)
            
        training_loop(args, fold_index, model, scoring_train_iterator, scoring_valid_iterator)

        with torch.no_grad():
            # execute testing loop
            featurize_test_dataset(args, test_index_data, fold_index)
            test_iterator = get_dataset_loaders(os.path.join(args.processed_dataset_path, 'featurized_dataset', 'test.jsonl'), batch_size=args.score_batch_size)
            # enabling the inference flag
            args.inference = True
            model = EssayScoringModel(args).to(device)
            load_checkpoint(model, args)
            args.logger.info('=='*30)
            args.logger.info('testing the trained model performance')
            _, test_wks = evaluate_model(args, fold_index, model, test_iterator, set_type='test', inference=True)
            test_kappa_score_list.append(test_wks)
            # finally disable the inference flag
            args.inference = False
        
    
    args.logger.info('**'*30)
    # finally logging the kappa scores obtained after k-fold training and testing
    args.logger.info('>>>> %d-fold quadratic kappa score : %f [%f]' % (args.kfolds, np.mean(test_kappa_score_list), np.std(test_kappa_score_list)))
    args.logger.info('kappa scores : %s' % test_kappa_score_list)
    wandb.log({'%d-fold_mean_kappa_score'%args.kfolds: np.mean(test_kappa_score_list),
                '%d-fold_kappa_score_std'%args.kfolds: np.std(test_kappa_score_list)})

if __name__ == "__main__":
    parser = ArgumentParser()
    default_checkpoint_path = os.path.join(currentdir, 'checkpoints')
    default_dataset_path = os.path.join(parentdir, 'processed_data')

    # Global model configuration
    parser.add_argument("--processed_dataset_path", type=str,
                        help="directory containing processed datasets", default=default_dataset_path)
    parser.add_argument('--checkpoint_path', default=default_checkpoint_path, type=str,
                        help='directory where checkpoints are stored')
    parser.add_argument('--rank_epochs', default=10, type=int,
                        help='number of total sentence ranking epochs to run')
    parser.add_argument('--rank_batch_size', default=1, type=int,
                        help='adjust essay ranking batch size per gpu')
    parser.add_argument('--nsamples', default=3, type=int,
                        help='specify samples to draw for essay ranking task.')
    parser.add_argument("--margin", default=1.0, type=float,
                        help="margin to use in pairwise sentence ranking loss.")
    parser.add_argument('--rank_learning_rate', default=1e-6, type=float, 
                        help='specify the learning rate for essay ranker.')
    parser.add_argument('--enable_coherence_signal', default=1, type=int, 
                        help='train the sentence ranker if it set to greater than 0')
    parser.add_argument('--coherence_vector', type=str,
                        help="specify the coherence vector file.")
    parser.add_argument('--score_epochs', default=10, type=int,
                        help='number of total sentence scoring epochs to run')
    parser.add_argument('--score_batch_size', default=4, type=int,
                        help='adjust essay scoring batch size per gpu')
    parser.add_argument('--score_clip_norm', type=float, default=1.0,
                        help="clip the gradient above the specified L2 norm")
    parser.add_argument('--score_weight_decay', default=0.1, type=float, 
                        help='specify the weight decay.')
    parser.add_argument('--score_learning_rate', default=1e-5, type=float, 
                        help='specify the learning rate for essay scorer')
    parser.add_argument('--weight_decay', default=0.01, type=float, 
                        help='specify the weight decay for sentence ranker.')
    parser.add_argument('--dropout_rate', default=0.1, type=float, 
                        help='specify the dropout rate for all layer, also applies to all transformer layers.')
    # parser.add_argument('--seed', default=42, type=int,
                        # help='seed value for random initialization.')
    parser.add_argument("--enable_scheduler", action='store_true',
                        help='activates the linear decay scheduler.')
    parser.add_argument("--warmup_steps", default=0.01, type=float,
                        help="percentage of total step used as tlinear warmup while training the model.")
    # dataset related configurations
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help="specify the maximum sequence length for processed dataset. \
                            if set to -1 then maximum length is obtained from corpus.")
    parser.add_argument('--max_fact_count', type=int, default=50, 
                        help="specify the max number of fact for each document. \
                            if set to -1 all facts will be considered")
    parser.add_argument('--max_fact_seq_len', type=int, default=50,
                        help="specify length of the concatenated fact for each doc. \
                            if set to -1 then maximum length is obtained from fact string")
    # logger configs
    parser.add_argument('--online_mode', default=0, type=int,
                        help='disables weight and bias syncronization if 0 is passed')
    # transformer config
    parser.add_argument('--arch', type=str, choices = [x for x in architecture_func], default='vanilla',
                        help='specify architecture type for sentence ranking module')
    parser.add_argument('--mtl_base_arch', type=str, default='vanilla', choices=['vanilla', 'hierarchical', 'fact-aware'],
                        help='specify the base architecture to be used with mtl approach.')
    # if "combined" architecture is active, we can disable mtl (on textual entailment data) associated with it
    parser.add_argument('--disable_mtl', type=int, default=0,
                        help='disable training on textual entailment dataset if set to greater than zero.')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        help='specify pretrained transformer model to use.')
    parser.add_argument('--tf2_model_name', type=str, default='roberta-base',
                        help='specify upper transformer model for fact-aware and hierarchical architecture.')
    parser.add_argument('--use_pretrained_tf2', type=int, default=0,
                        help='loads pretrained transformer model for TF2 layer for fact-aware and hierarchical architecture.')
    parser.add_argument('--sentence_pooling', type=str, default='max', choices = ['sum', 'mean', 'max', 'min', 'attention', 'none'],
                        help='specify the pooling strategy to use at lower transformer i.e. TF1 layer')
    parser.add_argument('--freeze_emb_layer', action='store_true',
                        help='freezes gradients updates at the embedding layer of lower transformer model.')
    parser.add_argument('--kfolds', type=int, default=5,
                        help='specify number of folds to use.')
    parser.add_argument('--prompt_id', type=int, default=1, choices=[x for x in range(9)],
                        help="specify prompt_id to use for training and testing on aes dataset. '0' mean collective training.")
    parser.add_argument('--enable_huber_loss', type=int, default=0,
                        help='enable huber loss as the regression loss.')
    parser.add_argument('--clip_norm', type=float, default=0,
                        help="clip the gradient above the specified L2 norm for sentence ranker")
    args  = parser.parse_args()

    full_arch_name = args.arch
    if args.arch=='mtl':
        full_arch_name="%s-%s"%(args.arch, args.mtl_base_arch)

    logger_exp_name = ("%s-wsj-v2-%s-prompt-%d"%(full_arch_name, args.model_name, args.prompt_id)).replace('/', '-')
    args.logger = MyLogger('', os.path.join(currentdir, "%s.log"%logger_exp_name), 
                            use_stdout=True, log_level=LOG_LEVELS.DEBUG)
    #get the arguments passed to this program
    params = {}
    for arg in vars(args):
        if arg in ["online_logger", "logger"]:
            continue
        params[arg] = getattr(args, arg)
    
    logger_args = {
        'project': 'correct_split_essay_scoring',    # first create a project on weight & bias with local account 
        'name': logger_exp_name,
        'config' : params,
        'tags' :['pytorch'],
    }
    
    # turn off the online sync
    if args.online_mode==0:
        logger_args.update({'mode': 'offline'}),
    
    # configure and add online logger
    wandb.init(**logger_args)
    
    # get the arguments passed to this program
    args.logger.info('\ncommand line argument captured ..')
    args.logger.info('--'*30)
    
    for key, value in params.items():
        args.logger.info('%s - %s' % (key, value))
    args.logger.info('--'*30)
    
    args.model_save_path = os.path.join(args.checkpoint_path, logger_exp_name)
    
    args.inference = False
    os.makedirs(args.model_save_path, exist_ok=True)
    # finally start the training and test iteration
    init(args)
    # uploading offline logger to wandb
    wandb.save(os.path.join(currentdir, "%s.log"%logger_exp_name))
