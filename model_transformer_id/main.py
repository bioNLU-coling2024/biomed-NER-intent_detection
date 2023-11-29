import pandas as pd
import numpy as np
import torch
import argparse
import json
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
    logging,
)

set_seed(41)

class dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        label_id = labels_to_ids[self.labels[idx]]
        item['labels'] = torch.tensor(label_id)
        return item

    def __len__(self):
        return len(self.labels)

    
def compute_metrics(p: EvalPrediction):
    preds_list, out_label_list = p.predictions.argmax(-1), p.label_ids        
    acc = accuracy_score(out_label_list, preds_list)
    report = classification_report(out_label_list, preds_list)
    return {'accuracy': acc, "report":report}


def prepare_dataset(dataset):
    if dataset == "mergedNER":
        df_train = pd.read_csv("ID_datasets/ner_merged_intent_train.csv")
        df_test = pd.read_csv("ID_datasets/ner_merged_intent_test.csv")
    else:
        raise NotImplemented
    
    A = df_train.label.unique()
    A = A[~pd.isnull(A)]
    global labels_to_ids, ids_to_labels
    labels_to_ids = {k: v for v, k in enumerate(A)}
    ids_to_labels = {v: k for v, k in enumerate(A)}
    
    return df_train, df_test

def train_model(df_train, df_test, model, epochs=10, batch_size=16, base_dir='./', max_len=512, savedir='./'):
    if (model in roberta_based_models):
        tokenizer = AutoTokenizer.from_pretrained(model,use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model,use_fast=True)
    train_encoding = tokenizer(df_train['sentence'].tolist(), padding='max_length', truncation=True, max_length=max_len)
    eval_encoding = tokenizer(df_test['sentence'].tolist(), padding='max_length', truncation=True, max_length=max_len)

    train_dataset = dataset(train_encoding, df_train['label'].tolist())
    eval_dataset = dataset(eval_encoding, df_test['label'].tolist())

    num_steps = len(train_dataset) * epochs // batch_size
    warmup_steps = num_steps // 10 
    
    training_args = TrainingArguments(
        output_dir = f'{base_dir}/models/',          
        num_train_epochs = epochs,              
        per_device_train_batch_size = batch_size,  
        per_device_eval_batch_size = batch_size,   
        logging_dir = f'{base_dir}/logs/',            
        evaluation_strategy = 'epoch',
        save_strategy="no",
        disable_tqdm= False,
        warmup_steps = warmup_steps,   
    )

    config = AutoConfig.from_pretrained(
        model,
        num_labels = len(labels_to_ids),
        id2label=ids_to_labels,
        label2id= labels_to_ids
    )    

    model = AutoModelForSequenceClassification.from_pretrained(model, config=config)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      compute_metrics=compute_metrics)

    trainer.train()
    trainer.save_model(savedir)
    
model_dict = {'bert': 'bert-base-uncased', 
                'roberta': 'roberta-base',
                'pubmedbert':'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER on BioMedical Datasets')
    parser.add_argument('-m', '--model', default='bert', type = str)
    parser.add_argument('-x', '--dataset', default='mergedNER', type = str)
    parser.add_argument('-e', '--epochs', default=6, type=int)
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-l', '--max_len', default=512, type=int)
    parser.add_argument('-v', '--verbose_setting', default=0, type=int)
    parser.add_argument('-d', '--base_dir', default='./', type = str)

    args = parser.parse_args()

    if args.verbose_setting:
        logging.set_verbosity_debug()
    df_train, df_test = prepare_dataset(args.dataset)

    roberta_based_models = ['allenai/biomed_roberta_base', 'roberta-base']    
    train_model(df_train, df_test, model_dict[args.model], args.epochs, args.batch_size, args.base_dir, args.max_len)