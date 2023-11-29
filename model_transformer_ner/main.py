import pandas as pd
import numpy as np
import torch
import argparse
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import Dataset

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
    logging,
)

set_seed(42)

FILL_TOKEN = '-100'

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        sentence = self.data.sentence[index].strip().split()  
        word_labels = self.data.word_labels[index].split(",") 

        encoding = self.tokenizer(sentence,
                             is_split_into_words=True, 
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)

        labels = [labels_to_ids[label] for label in word_labels] 
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = labels[i]
                i += 1

        item = {key: torch.as_tensor(val) for key, val in encoding.items() if key!='offset_mapping'}
        item['labels'] = torch.as_tensor(encoded_labels)

        return item
        
    def __len__(self):
        return self.len

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray):
    preds = np.argmax(predictions, axis=2)      

    batch_size, seq_len = preds.shape
    
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(ids_to_labels[label_ids[i][j]])
                preds_list[i].append(ids_to_labels[preds[i][j]])
    
    return preds_list, out_label_list
    
def compute_metrics(p: EvalPrediction):
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        
    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

def get_df(path):
    texts, labels = [], []
    text, label = [], []
    with open(path, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
        for line in lines:
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                text.append(word)
                label.append(pairs[-1])
            else:
                if len(text) > 0:
                    texts.append(text)
                    labels.append(label)
                text, label = [], []
    texts_joined = [' '.join(i) for i in texts]
    labels_joined = [','.join(i) for i in labels]
    
    df = pd.DataFrame()
    df['sentence'] = texts_joined
    df['word_labels'] = labels_joined
    A = list(set([x for y in labels for x in y]))
    return df, A

def prepare_dataset(dataset):
    if dataset == 'DDI':
        df_train_medline, A1 = get_df("NER_datasets/train_medline.txt")
        df_train_drugbank, A2 = get_df("NER_datasets/train_drugbank.txt")
        df_test_medline, _ = get_df("NER_datasets/test_medline.txt")
        df_test_drugbank, _ = get_df("NER_datasets/test_drugbank.txt")
        df_train = df_train_medline.append(df_train_drugbank).reset_index(drop=True)
        df_test = df_test_medline.append(df_test_drugbank).reset_index(drop=True)
        A = list(set(A1+A2))
        
    elif dataset in ['jnlpba', 'bc5cdr', 'ncbi', 'anatem']:
        df_train, A = get_df("NER_datasets/"+dataset+"/train_"+dataset +".txt")
        df_test,_ = get_df("NER_datasets/"+dataset+"/test_"+dataset +".txt")
    else:
        raise NotImplementedError
    
    global labels_to_ids, ids_to_labels
    labels_to_ids = {k: v for v, k in enumerate(A)}
    ids_to_labels = {v: k for v, k in enumerate(A)}

    return df_train, df_test

def train_model(df_train, df_test, model, epochs=15, batch_size=16, base_dir='./', max_len=512, model_type = 'bert', savedir='./'):

    if (model_type == 'roberta') or (model in roberta_based_models):
        tokenizer = AutoTokenizer.from_pretrained(model,use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model,use_fast=True)
        
    train_dataset = dataset(df_train, tokenizer, max_len)
    eval_dataset = dataset(df_test, tokenizer, max_len)
    
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

    model = AutoModelForTokenClassification.from_pretrained(model, config=config)

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=eval_dataset,
                      compute_metrics=compute_metrics)

    trainer.train()
    
    trainer.save_model(savedir)

    
model_dict = {'bert': 'bert-base-uncased', 'roberta': 'roberta-base', 
                    'biobert':'dmis-lab/biobert-base-cased-v1.1', 
                    'pubmedbert':'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                    'biomedroberta':'allenai/biomed_roberta_base',
                    'clinicbert':'emilyalsentzer/Bio_ClinicalBERT'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER on BioMedical Datasets')
    parser.add_argument('-m', '--model', default='bert', type = str)
    parser.add_argument('-x', '--dataset', default='jnlpba', type = str)
    parser.add_argument('-t', '--model_type',default='roberta', type = str)
    parser.add_argument('-e', '--epochs', default=35, type=int)
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-l', '--max_len', default=512, type=int)
    parser.add_argument('-d', '--base_dir', default='./', type = str)
    parser.add_argument('-s', '--savedir', default='./', type = str)

    args = parser.parse_args()

    df_train, df_test = prepare_dataset(args.dataset)

    roberta_based_models = ['allenai/biomed_roberta_base', 'roberta-base']
    
    train_model(df_train, df_test, model_dict[args.model], args.epochs, args.batch_size, args.base_dir, args.max_len, args.model_type, args.savedir)
