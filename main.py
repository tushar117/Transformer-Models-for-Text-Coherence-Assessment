import pytorch_lightning as pl
from argparse import ArgumentParser
import os
import numpy as np
import torch
import random
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import transformers
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.metrics import Metric
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_recall_fscore_support
import json
from logger import MyLogger, LOG_LEVELS
from models import *
from pytorch_lightning.loggers import WandbLogger
import sys
from pytorch_lightning.utilities import rank_zero_only
from dataset_processor.featurize_dataset import GCDCFeaturizer, WSJFeaturizer, MTLFeaturizer, CombinedFeaturizer
from scipy.stats import spearmanr
from data_loader import get_dataset_loaders
from utils.common import normalize_gcdc_sub_corpus


base_dir = os.path.dirname(os.path.realpath(__file__))

def get_allowed_operations():
    # reflect the change done here to the file: ./dataset_processor/featurize_dataset.py
    dataset_options = {
        "wsj": {
            "tasks" : ["sentence-ordering"],
        },
        "gcdc": {
            "tasks" : ['3-way-classification', 'minority-classification', 'sentence-ordering', 'sentence-score-prediction'],
            "sub_corpus" : ['All', 'Clinton', 'Enron', 'Yelp', 'Yahoo'],
        }
    }
    return dataset_options

#allow deterministic psuedo-random-initialization
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

class PairWiseSentenceRanking(nn.Module):
    """Head for pairwise sentence ranking task"""
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(PairWiseSentenceRanking, self).__init__()
        #document transformation
        self.phi = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(hidden_dropout_prob)
    
    def forward(self, sent_a, sent_b):
        return self.phi(self.dropout(sent_a)), self.phi(self.dropout(sent_b))

class SentenceScorer(nn.Module):
    "Head for sentence score task (regression)"
    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, features):
        x = features  # takes [CLS] token representation as input
        x = self.dropout(x)
        x = self.dense(x)
        return x

class ModelWrapper(pl.LightningModule):
    def __init__(self, args):
        super(ModelWrapper, self).__init__()
        self.config_args = args
        #load pretrained hierarchical model
        self.doc_encoder = get_module(args.arch, base_arch=args.mtl_base_arch)(args)
        
        #task specific model handlers
        if self.config_args.task == "sentence-ordering":
            self.task_head = PairWiseSentenceRanking(self.doc_encoder.tf2.config.hidden_size, args.dropout_rate)
            #metrics
            self.train_metric = pl.metrics.Accuracy()
            self.val_metric = pl.metrics.Accuracy()
            self.test_metric = pl.metrics.Accuracy()
        elif self.config_args.task == "3-way-classification":
            self.task_head = TexClassificationHead(self.doc_encoder.tf2.config.hidden_size, 3, args.dropout_rate)
            #metrics
            self.train_metric = pl.metrics.Accuracy()
            self.val_metric = pl.metrics.Accuracy()
            self.test_metric = pl.metrics.Accuracy()
        elif self.config_args.task == "minority-classification":
            self.task_head = TexClassificationHead(self.doc_encoder.tf2.config.hidden_size, 2, args.dropout_rate)
            # metric
            # calculates F0.5 score at the inference time and while training use accuracy metric
            self.train_metric = pl.metrics.Accuracy()
            self.val_metric = pl.metrics.Accuracy()
            self.test_metric = pl.metrics.FBeta(num_classes=2, beta=0.5, average=None)
        elif self.config_args.task == "sentence-score-prediction":
            self.task_head = SentenceScorer(self.doc_encoder.tf2.config.hidden_size, args.dropout_rate)
        
        # handling the multi-task learning approach
        if self.config_args.arch in ['mtl', 'combined']:
            # adding textual entailment task head
            self.te_task_head = TexClassificationHead(self.doc_encoder.tf2.config.hidden_size, 2, args.dropout_rate)
            # adding accuracy metrics as well 
            self.te_train_metric = pl.metrics.Accuracy()
            self.te_val_metric = pl.metrics.Accuracy()

    def _get_task_specific_dataset_index(self, dataset_ids):
        entailment_data_idx = []
        task_specific_data_idx = []
        for idx, d_id in enumerate(dataset_ids):
            # fetching the indexes for different tasks
            if d_id == 0:
                entailment_data_idx.append(idx)
            else:
                task_specific_data_idx.append(idx)
        return entailment_data_idx, task_specific_data_idx

    def forward(self, input_data):
        output_logits = []
        # dummy indicator (when MTL is inactive) variable for task specific data instances
        task_specific_data_idx = [i for i in range(input_data[0].shape[0])]

        if self.config_args.arch in ['mtl', 'combined']:
            # gather textual entailment dataset i.e. d_id == 0
            d_ids = input_data[0]             
            entailment_data_idx, task_specific_data_idx = self._get_task_specific_dataset_index(d_ids)
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
            if self.config_args.task == "sentence-ordering":
                assert len(input_data)%2 == 0
                mid = int(len(input_data)/2)
                doc_a_data = input_data[:mid]
                doc_b_data = input_data[mid:]
                coherence_a = self.doc_encoder(doc_a_data)
                coherence_b = self.doc_encoder(doc_b_data)
                output_logits.append(self.task_head(coherence_a, coherence_b))
            else:
                doc_rep = self.doc_encoder(input_data)
                output_logits.append(self.task_head(doc_rep))
        
        return output_logits

    def configure_optimizers(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config_args.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.config_args.learning_rate, eps=1e-8)
        # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.config_args.learning_rate, eps=1e-6)
        
        if self.config_args.enable_scheduler:
            total_dataset_count = self.config_args.train_dataset_count
            total_steps = int(np.ceil((self.config_args.epochs * total_dataset_count)/(self.config_args.batch_size*self.config_args.gpus)))
            
            scheduler = {
                # 'scheduler': get_constant_schedule_with_warmup(optimizer, self.config_args.warmup_steps*total_steps)
                'scheduler': get_linear_schedule_with_warmup(optimizer, self.config_args.warmup_steps*total_steps, total_steps),
                'interval': 'step',
            }
            return [optimizer], [scheduler]

        return optimizer
    
    def _step(self, batch, step_type):
        input_data, label_ids = batch[:-1], batch[-1]
        # dummy indicator (when MTL is inactive) variable for task specific data instances
        task_specific_data_idx = [i for i in range(label_ids.view(-1).size(0))]
        model_outputs = self(input_data)
        loss = 0
        task_loss = 0

        res = {}
        pbar = {}
        online_logger_data = {}
        # select the metric
        loss_label = '%s_loss' % step_type if step_type!='train' else 'loss'
        
        if self.config_args.task != "sentence-score-prediction":
            if step_type == 'train':
                step_metric = self.train_metric
            elif step_type == 'val':
                step_metric = self.val_metric
            else:
                step_metric = self.test_metric

        # architecture specific (MTL) loss calculation
        if self.config_args.arch in ["mtl", "combined"] and step_type != "test":
            d_ids = input_data[0]
            entailment_data_idx, task_specific_data_idx = self._get_task_specific_dataset_index(d_ids)
            if len(entailment_data_idx):
                te_pred = model_outputs[0]
                entailment_label_ids = label_ids[entailment_data_idx].long()
                te_loss = F.cross_entropy(te_pred, entailment_label_ids)
                te_step_metric = getattr(self, "te_%s_metric"%step_type)
                te_acc = te_step_metric(te_pred.softmax(dim=-1), entailment_label_ids)
                pbar['batch_%s_te_acc'%step_type] = te_acc
                online_logger_data.update({'batch_%s_te_loss' % step_type: te_loss })
                loss += te_loss
                res["%s_te_loss" % step_type] = te_loss
            online_logger_data.update({"te_data_count": len(entailment_data_idx),
                                       "task_data_count": len(task_specific_data_idx)})
            # assigning the task_specific_labels to label_ids variable for normal flow
            label_ids = label_ids[task_specific_data_idx]

        if len(task_specific_data_idx):
            # task specific loss calculation
            model_output = model_outputs[-1]
            if self.config_args.task == "sentence-ordering":
                # sentence score for 'sentence a' and 'sentence b' are 'phi_a' and 'phi_b' respectively
                phi_a, phi_b = model_output
                task_loss = F.margin_ranking_loss(phi_a, phi_b, label_ids, margin=self.config_args.margin)
                pred_res = -1*torch.ones_like(label_ids)
                pred_res[phi_a.detach().requires_grad_(False).view(-1) > phi_b.detach().requires_grad_(False).view(-1)] = 1
                # get batch accuracy       
                acc = step_metric(pred_res.long(), label_ids.long())
                pbar['batch_%s_acc'%step_type] = acc
            elif self.config_args.task == "3-way-classification":
                if self.config_args.enable_kldiv:
                    #experimenting with KL divergence loss
                    num_classes = 3
                    smoothing = self.config_args.label_smoothing
                    smoothed_labels = (smoothing/(num_classes - 1))*torch.ones_like(model_output.view(-1, num_classes))
                    smoothed_labels.scatter_(1, label_ids.unsqueeze(1), 1-smoothing)
                    #assert smoothed_labels.sum() == len(label_ids)
                    task_loss = F.kl_div(model_output.view(-1, num_classes).log_softmax(dim=-1), smoothed_labels, reduction='batchmean', log_target=False)
                else:
                    task_loss = F.cross_entropy(model_output, label_ids.long())
                # get training batch accuracy
                acc = step_metric(model_output.softmax(dim=-1), label_ids.long())
                pbar['batch_%s_acc'%step_type] = acc
            elif self.config_args.task == "minority-classification":
                if self.config_args.enable_kldiv:
                    #experimenting with KL divergence loss
                    num_classes = 2
                    smoothing = self.config_args.label_smoothing
                    smoothed_labels = (smoothing/(num_classes - 1))*torch.ones_like(model_output.view(-1, num_classes))
                    smoothed_labels.scatter_(1, label_ids.unsqueeze(1), 1-smoothing)
                    #assert smoothed_labels.sum() == len(label_ids)
                    task_loss = F.kl_div(model_output.view(-1, num_classes).log_softmax(dim=-1), smoothed_labels, reduction='batchmean', log_target=False)
                else:
                    task_loss = F.cross_entropy(model_output, label_ids.long())
                # get training batch accuracy
                if step_type == "test":
                    acc = step_metric(model_output.softmax(dim=-1), label_ids.long())[1]
                else:
                    acc = step_metric(model_output.softmax(dim=-1), label_ids.long())
                pbar['batch_%s_acc'%step_type] = acc
            elif self.config_args.task == "sentence-score-prediction":
                task_loss = F.mse_loss(model_output.view(-1), label_ids.view(-1))
                res.update({'preds': model_output, 'labels': label_ids})
            
            if self.config_args.arch in ["mtl", "combined"] and step_type != "test":
                online_logger_data.update({'batch_%s_task_loss' % step_type: task_loss})
        
        loss += task_loss
        online_logger_data.update(pbar)
        online_logger_data.update({'batch_%s_loss'%step_type: loss})
        self.logger.log_metrics(online_logger_data)

        res[loss_label] = loss
        if len(pbar):
            res['progress_bar'] = pbar
        
        return res

    def _epoch_end(self, step_outputs, end_type):
        # select the metric
        loss_label = '%s_loss' % end_type if end_type!="train" else "loss"
        
        if self.config_args.task != "sentence-score-prediction":
            if end_type == 'train':
                end_metric = self.train_metric
            elif end_type == 'val':
                end_metric = self.val_metric
            else:
                end_metric = self.test_metric

        avg_loss = torch.tensor([x[loss_label] for x in step_outputs]).mean()
        #task specific model handlers
        if self.config_args.task=='sentence-score-prediction':
            #handling sentence-score-prediction task separately
            labels = torch.cat([x['labels'] for x in step_outputs if 'labels' in x] ).view(-1)
            preds = torch.cat([x['preds'] for x in step_outputs if 'preds' in x]).view(-1)
            s_corr, pvalue = spearmanr(labels.detach().cpu().numpy(), preds.detach().cpu().numpy())
            overall_acc = torch.tensor(s_corr)
            self.config_args.logger.info('epoch : %d - average_%s_loss : %f, overall_%s_spearman_score : %f' % (self.current_epoch, end_type, avg_loss.item(), 
                                                                                                end_type, overall_acc.item()))
        else: 
            if self.config_args.task=="minority-classification" and end_type=="test":
                fbeta = end_metric.compute()
                overall_acc = torch.tensor(fbeta[1])
            else:   
                overall_acc = end_metric.compute()
            self.config_args.logger.info('epoch : %d - average_%s_loss : %f, overall_%s_acc : %f' % (self.current_epoch, end_type, avg_loss.item(), 
                                                                                                end_type, overall_acc.item()))

        # architecture specific (MTL) loss calculation
        if self.config_args.arch in ["mtl", "combined"] and end_type !='test':
            te_loss_label = "%s_te_loss"%end_type
            filtered_te_datapoints =  [x[te_loss_label] for x in step_outputs if te_loss_label in x]
            if len(filtered_te_datapoints):
                avg_te_loss = torch.tensor(filtered_te_datapoints).mean()
                overall_te_acc = getattr(self, "te_%s_metric"%end_type).compute()
                self.config_args.logger.info('epoch : %d - average_%s_te_loss : %f, overall_%s_te_acc : %f' % (self.current_epoch, end_type, avg_te_loss.item(), 
                                                                                                    end_type, overall_te_acc.item()))
                self.logger.log_metrics({'avg_%s_te_loss'%end_type: avg_te_loss, '%s_te_acc'%end_type: overall_te_acc})

        # logging to weight and bias if online mode is enabled
        self.logger.log_metrics({'avg_%s_loss'%end_type: avg_loss, '%s_acc'%end_type: overall_acc})
        self.log('avg_%s_loss'%end_type, avg_loss, prog_bar=True)
        self.log('overall_%s_acc'%end_type, overall_acc, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def training_epoch_end(self, train_step_outputs):
        self._epoch_end(train_step_outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')
    
    def validation_epoch_end(self, val_step_outputs):
        self._epoch_end(val_step_outputs, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def test_epoch_end(self, test_step_outputs):
        self._epoch_end(test_step_outputs, 'test') 

class TextDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def val_dataloader(self):
        dev_file_path = os.path.join(os.path.abspath(self.args.processed_dataset_path), 'featurized_dataset', 'dev.jsonl')
        val_dataset =  get_dataset_loaders(dev_file_path, self.args.val_dataset_count, batch_size=self.args.batch_size, regression=self.args.task=='sentence-score-prediction')
        return val_dataset

    def train_dataloader(self):
        train_file_path = os.path.join(os.path.abspath(self.args.processed_dataset_path), 'featurized_dataset', 'train.jsonl')
        train_dataset =  get_dataset_loaders(train_file_path, self.args.train_dataset_count, batch_size=self.args.batch_size, regression=self.args.task=='sentence-score-prediction')
        return train_dataset

def start_training(args):
    model_name = args.logger_exp_name

    args.logger.debug('initiating training process...')
    
    final_checkpoint_path = os.path.join(args.checkpoint_path, model_name)  
    os.makedirs(final_checkpoint_path, exist_ok=True)

    # Load datasets
    dm = TextDataModule(args)

    call_back_parameters = {
        'filepath': final_checkpoint_path,
        'save_top_k': 1,
        'verbose': True,
        'monitor': 'overall_val_acc',
        'mode': 'max',
    }

    # checkpoint callback to used by the Trainer
    checkpoint_callback = ModelCheckpoint(**call_back_parameters)

    model = ModelWrapper(args)
    
    args.logger.debug(model)
    args.logger.info('Model has %d trainable parameters' % count_parameters(model))

    callback_list = []
    
    precision_val = 16 if args.fp16 > 0 else 32

    trainer = pl.Trainer(callbacks=callback_list, max_epochs=args.epochs, min_epochs=1, gradient_clip_val=args.clip_grad_norm, 
                        gpus=args.gpus, checkpoint_callback=checkpoint_callback, distributed_backend='ddp', logger=args.online_logger,
                        precision=precision_val, auto_select_gpus=True)
    #finally train the model
    args.logger.debug('about to start training loop...')
    trainer.fit(model, dm)
    args.logger.debug('training done.')

def test_dataloader(args):
    test_file_path = os.path.join(os.path.abspath(args.processed_dataset_path), 'featurized_dataset', 'test.jsonl')
    test_dataset =  get_dataset_loaders(test_file_path, args.test_dataset_count, batch_size=args.batch_size, regression=args.task=='sentence-score-prediction')
    return test_dataset

def init_testing(args): 
    args.logger.debug('initiating inference process...')
    model = ModelWrapper(args)

    #load trained model
    args.logger.debug('loading the model from checkpoint : %s' % args.checkpoint_path)
    trained_model = model.load_from_checkpoint(args.checkpoint_path, args=args)
    args.logger.debug('loaded model successfully !!!')

    test_loader = test_dataloader(args)
        
    # invoke callbacks if required
    callback_list = []
    if args.corpus == 'gcdc':
        args.logger.info('testing on dataset : %s ( sub_dataset %s ) on task : %s' % (args.corpus, args.sub_corpus, args.task))
    else:
        args.logger.info('testing on dataset : %s ( sub_dataset %s ) on task : %s' % (args.corpus, args.sub_corpus, args.task))
    
    precision_val = 16 if args.fp16 > 0 else 32
    
    trainer = pl.Trainer(callbacks=callback_list, gpus=args.gpus, distributed_backend='ddp',logger=args.online_logger, 
                                precision=precision_val, auto_select_gpus=True)
    trainer.test(trained_model, test_dataloaders=test_loader)
    args.logger.debug('testing done !!!')

def dataset_count(args):
    # internal function that count the datainstance present in dataset
    def count_lines(file_path):
        count = 0
        with open(file_path, 'r', encoding='utf-8') as dfile:
            for line in dfile.readlines():
                count+=1
        return count

    if args.inference:
        args.test_dataset_count = count_lines(os.path.join(os.path.abspath(args.processed_dataset_path), 'featurized_dataset', 'test.jsonl')) if not args.test_dataset_count else args.test_dataset_count
    else:
        args.val_dataset_count = count_lines(os.path.join(os.path.abspath(args.processed_dataset_path), 'featurized_dataset', 'dev.jsonl')) if not args.val_dataset_count else args.val_dataset_count
        args.train_dataset_count = count_lines(os.path.join(os.path.abspath(args.processed_dataset_path), 'featurized_dataset', 'train.jsonl')) if not args.train_dataset_count else args.train_dataset_count

def check_config(config):
    dataset_options = get_allowed_operations()
    if config.corpus not in dataset_options:
        raise Exception("no handler module found for dataset : ' %s ' !!!" % config.corpus)

    if config.corpus=='gcdc' and config.sub_corpus.lower() not in [x.lower() for x in dataset_options['gcdc']['sub_corpus']]:
        raise Exception("invalid GCDC sub_corpus selected : ' %s ', please select one from %s" % (config.sub_corpus, 
                                                                                dataset_options['gcdc']['sub_corpus']))

    if config.task not in dataset_options[config.corpus]['tasks']:
        raise Exception("' %s ' invalid task selected for corpus : %s, please select one from %s" % (config.task, 
                                                                    config.corpus, dataset_options[config.corpus]['tasks']))
    
    if config.task == "sentence-score-prediction" and config.gpus != 1:
        # don't change this condition, otherwise we have to write your own distributed spearman scoring metric in pytorch lightning
        config.logger.warn('changing number of GPUs from %d to 1.' % config.gpus)
        config.gpus = 1
    
    if config.arch in ["mtl", 'combined'] and config.gpus != 1:
        # TODO: support fpr mutiple-GPU training for MTL and combined architecture. Currently, only single GPU training
        # is supported for theese architectures.
        config.logger.warn('[%s] changing number of GPUs from %d to 1.' % (config.arch, config.gpus))
        config.gpus = 1

if __name__ == "__main__":
    parser = ArgumentParser()

    default_checkpoint_path = os.path.join(base_dir, 'lightning_checkpoints')
    default_dataset_path = os.path.join(base_dir, 'processed_data')

    # Global model configuration
    parser.add_argument("--processed_dataset_path", type=str,
                        help="directory containing processed datasets", default=default_dataset_path)
    parser.add_argument('--checkpoint_path', default=default_checkpoint_path, type=str,
                        help='directory where checkpoints are stored')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--learning_rate', default=1e-6, type=float, 
                        help='specify the learning rate')
    parser.add_argument('--clip_grad_norm', default=0.0, type=float, 
                        help='clip gradients with norm above specified value, 0 value will disable it.')
    parser.add_argument('--weight_decay', default=0.01, type=float, 
                        help='specify the weight decay.')
    parser.add_argument('--dropout_rate', default=0.1, type=float, 
                        help='specify the dropout rate for all layer, also applies to all transformer layers.')
    # parser.add_argument('--seed', default=42, type=int,
                        # help='seed value for random initialization.')
    parser.add_argument("--enable_scheduler", action='store_true',
                        help='activates the linear decay scheduler.')
    parser.add_argument("--warmup_steps", default=0.01, type=float,
                        help="percentage of total step used as tlinear warmup while training the model.")
    parser.add_argument("--margin", default=1.0, type=float,
                        help="margin to use in pairwise sentence ranking loss.")
    # dataset related configurations
    parser.add_argument('--corpus', choices=['wsj', 'gcdc'], type=str, default='gcdc',
                        help="specify the corpus.")
    parser.add_argument('--sub_corpus', type=str, default='all',
                        help="specify the sub-corpus for the gcdc dataset.")
    # if "allenai/longformer-base-4096" used then pass 2048 as max_seq_len
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help="specify the maximum sequence length for processed dataset. \
                            if set to -1 then maximum length is obtained from corpus.")
    parser.add_argument('--max_fact_count', type=int, default=50, 
                        help="specify the max number of fact for each document. \
                            if set to -1 all facts will be considered")
    parser.add_argument('--max_fact_seq_len', type=int, default=50,
                        help="specify length of the concatenated fact for each doc. \
                            if set to -1 then maximum length is obtained from fact string")
    parser.add_argument('--permutation_count', type=int, default=20,
                        help="number of permutation allowed per data sample")
    parser.add_argument('--with_replacement', default=1, type=int,
                        help="if greater than 1, it will draw new samples with replacement")
    # below three arguments are for debugging purpose
    parser.add_argument("--train_dataset_count", type=int,
                        help="specify number of training data to use. (for debugging purpose)")
    parser.add_argument("--val_dataset_count", type=int,
                        help="specify number of validation data to use. (for debugging purpose)")
    parser.add_argument("--test_dataset_count", type=int,
                        help="specify number of testing data to use. (for debugging purpose)")
    # TODO: complete implementation of below parameter
    parser.add_argument('--inverse_pra', default=0, type=int,
                        help="enables inverse pairwise ranking accuracy at inference for sentence-ordering tasks.")
    # task related configuration
    parser.add_argument('--task', type=str, required=True,
                        help="specify the task to be performed on the selected dataset.")
    parser.add_argument('--enable_kldiv', action='store_true',
                        help="if gcdc corpus is selected, then use KL divergence loss instead of Cross entropy loss for \
                            3-way-classification and minority-classification.")
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help="lable smoothing for kl divergence loss.")
    # inference
    parser.add_argument('-i', '--inference', action='store_true',
                        help='enable inference over the datasets')
    # logger configs
    parser.add_argument('--online_mode', default=0, type=int,
                        help='disables weight and bias syncronization if 0 is passed')
    parser.add_argument('--logger_exp_name', type=str, 
                        help='specified name will be used to store checkpoints.')
    # transformer config
    parser.add_argument('--arch', type=str, choices = [x for x in architecture_func], required=True,
                        help='specify type of architecture')
    # if "combined" architecture is active, we can disable mtl (on textual entailment data) associated with it
    parser.add_argument('--disable_mtl', type=int, default=0,
                        help='disable training on textual entailment dataset if set to greater than zero.')
    parser.add_argument('--mtl_base_arch', type=str, default='vanilla', choices=['vanilla', 'hierarchical', 'fact-aware'],
                        help='specify the base architecture to be used with mtl approach.')
    parser.add_argument('--model_name', type=str, default='roberta-base',
                        help='specify pretrained transformer model to use.')
    parser.add_argument('--tf2_model_name', type=str, default='roberta-base',
                        help='specify upper transformer model for fact-aware and hierarchical architecture.')
    parser.add_argument('--use_pretrained_tf2', type=int, default=0,
                        help='loads pretrained transformer model for TF2 layer for fact-aware and hierarchical architecture.')
    parser.add_argument('--sentence_pooling', type=str, default='none', choices = ['sum', 'mean', 'max', 'min', 'attention', 'none'],
                        help='specify the pooling strategy to use at lower transformer i.e. TF1 layer')
    parser.add_argument('--freeze_emb_layer', action='store_true',
                        help='freezes gradients updates at the embedding layer of lower transformer model.')
    # experimental run count
    parser.add_argument('--exp_count', type=int, default=0,
                        help='specify the experiment run id when executing mutiple iterations of same experiment.')
    # GPU memory utilization optimizations
    parser.add_argument('--fp16', type=int, default=0,
                        help='enable the automatic mixed precision training')
    args  = parser.parse_args()
    
    full_arch_name = args.arch
    if args.arch=='mtl':
        full_arch_name="%s-%s"%(args.arch, args.mtl_base_arch)
    if not args.logger_exp_name:
        corpus_name = args.corpus
        if args.corpus.lower()=='gcdc':
            corpus_name = "%s-%s"%(corpus_name, normalize_gcdc_sub_corpus(args.sub_corpus))
        args.logger_exp_name = "%s-%s-%s-%s"%(corpus_name, full_arch_name, args.task, args.model_name)
        if args.exp_count!=0:
            args.logger_exp_name = "%s-%d"%(args.logger_exp_name, args.exp_count)

    # replacing os.path separator if exists in the logger_exp_name
    args.logger_exp_name = args.logger_exp_name.replace('/', '-')

    # random_seed(args.seed)
    overwrite_flag = False if args.inference else True
    # offline logger
    args.logger = MyLogger('', os.path.join(base_dir, "%s.log"%args.logger_exp_name), 
                            use_stdout=True, log_level=LOG_LEVELS.DEBUG, overwrite=overwrite_flag)

    # sanity check of the tasks and dataset related configurations
    check_config(args)
    # featurize the task specific dataset for selected corpus only one time
    if rank_zero_only.rank == 0:
        # this will enforce to run preprocessing step during the process start on zeroth GPU and store it to common location.
        # It can be later loaded at each GPU in multi-gpu settings
        featurizer = GCDCFeaturizer if args.corpus == 'gcdc' else WSJFeaturizer
        featurizer(args).featurize_dataset(inference=args.inference)
        # merge textual entailment dataset if architecture is MTL
        if args.arch=='mtl':
            mtl_featurizer = MTLFeaturizer(args)
            mtl_featurizer.featurize_dataset(inference=args.inference)
        
        if args.arch=='combined':
            combined_featurizer = CombinedFeaturizer(args)
            combined_featurizer.featurize_dataset(inference=args.inference)
    
    dataset_count(args)
    
    #get the arguments passed to this program
    params = {}
    for arg in vars(args):
        if arg in ["online_logger", "logger"]:
            continue
        params[arg] = getattr(args, arg)
    
    logger_args = {
        'project': 'dry_run_text_coherence',    # first create a project on weight & bias with local account 
        'name': args.logger_exp_name,
        'config' : params,
        'tags' :['pytorch-lightning'],
    }
    
    # updates the logger name if inference is active
    if args.inference:
        logger_args.update({
            'name': args.logger_exp_name+'-test',
            })
    # turn off the online sync
    if args.online_mode==0:
        logger_args.update({'offline': True}),
    
    # configure and add logger to arguments
    args.online_logger = WandbLogger(**logger_args)
    
    # get the arguments passed to this program
    args.logger.info('\ncommand line argument captured ..')
    args.logger.info('--'*30)
    
    for key, value in params.items():
        args.logger.info('%s - %s' % (key, value))
    args.logger.info('--'*30)

    if args.inference:
        init_testing(args)
    else:
        start_training(args)
