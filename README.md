# Transformer Models for Text Coherence Assessment

We investigate four different Transformer-based architectures for the different discourse coherence task: vanilla Transformer, hierarchical Transformer, multi-task learning based model and a model with fact based input representation. One can find more details, analyses, and baseline results in [our paper](https://arxiv.org/abs/2109.02176). 

This implementation is based on PyTorch 1.6.0 with Python 3.6.10.

### Installation

Install the required packages:

```
pip install -r requirements.txt
```

### Dataset

Before training or evaluating model, download the dataset from [here](https://drive.google.com/file/d/1ySrFSIPcY5r19pGja6SQ7jCsUkYsoMLc/view?usp=sharing) and unzip within the project directory.

It includes Grammarly Corpus of Discourse Coherence (GCDC), Wall Street Journal (WSJ) and Recognizing Textual Entailment (RTE) datasets

Different discouse coherence tasks defined for the datasets mentioned above:

- GCDC
    - **3-way-classification** : Given the document, the task is to classify it into 3 different label - high, medium and low on which denotes the textual coherence 
level of the given document.
    - **minority-classification** : A binary classification task in which dataset is created from by labeling low coherent text through majority voting (if 2 expert agrees out of 3) and not low coherence otherwise.
    - **sentence-ordering** : The document which are labelled with high coherence tags (in GCDC) are taken and 20 random permutation is obtained for each document. The dataset is created by pairing the original setence and permutated sentence. The task is of choosing original document from original-permuted pair.
    - **sentence-score-prediction** : The coherence labels from 3 different experts are averaged out to create a gold score and the task is to predict this score. (regression task).
- WSJ
    - **sentence-ordering** : All the document present in WSJ except the document containing one sentence are taken and 20 random permutation is obtained  for each document. The dataset is created by pairing the original setence and permutated sentence. Task is of choosing original document from original-permuted pair.

### Training and Evaluation manually

To train different transformer based architecture models:

```
python main.py --arch <ARCH> --epochs <EPOCHS> --gpus <GPUS> --batch_size <BATCH_SIZE> --learning_rate <LR> --corpus <CORPUS> --sub_corpus <SUB_CORPUS> --model_name <MODEL_NAME> --freeze_emb_layer --online_mode <ONLINE_MODE> --task <TASK>
```

where,

- `corpus` can take one of 'gcdc' or 'wsj'.
- `sub_corpus` can take anyone value from 'All', 'Clinton', 'Enron', 'Yelp' or 'Yahoo' given that `corpus` is 'gcdc' 
- `arch` can take one of 'vanilla', 'hierarchical', 'mtl' or 'fact-aware'
- `task` can take one of '3-way-classification', 'minority-classification', 'sentence-ordering' or 'sentence-score-prediction'for GCDC dataset and only 'sentence-ordering' for WSJ dataset
- `model_name` defines transformer model to use. (by-default its's roberta-base)

To evaluate the trained model over test dataset:

```
python main.py --inference --arch <ARCH> --gpus <GPUS> --batch_size <BATCH_SIZE> --corpus <CORPUS> --sub_corpus <SUB_CORPUS> --model_name <MODEL_NAME> --freeze_emb_layer --online_mode <ONLINE_MODE> --task <TASK>'
```

### Training and Evaluation through automated script

Whole pipeline can be executed using the following bash file. User need to change the variables accordingly as mentioned in bash script.

```
./run.sh
```
### Citation
One can cite it as follows:

```
@article{abhishek2021transformer,
  title={Transformer models for text coherence assessment},
  author={Abhishek, Tushar and Rawat, Daksh and Gupta, Manish and Varma, Vasudeva},
  journal={arXiv preprint arXiv:2109.02176},
  year={2021}
}
```
