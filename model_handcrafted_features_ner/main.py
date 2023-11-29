'''
This code was adopted from (https://github.com/melanietosik/maxent-ner-tagger)
'''

import pickle
import sys
import numpy as np
import pandas as pd
import spacy
import argparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from seqeval.metrics import f1_score
from spacy.tokens import Doc

class CustomTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, words):
        return Doc(self.vocab, words=words)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = CustomTokenizer(nlp.vocab)

MODEL_FP = "model.pickle"
VECTORIZER_FP = "vectorizer.pickle"
FEATURE_NAMES_FP = "features.txt"

class FeatureBuilder:

    def __init__(self, sent, feat):

        self.sent = sent  
        self.feat = feat  
        self.doc = nlp(sent)
        self.use = ["is_title", "orth_", "lemma_", "lower_", "norm_", "shape_", "prefix_", "suffix_",]

        assert(len(self.sent) == len(self.doc))

        for x in self.use:
            for tok in self.doc:
                getattr(self, x)(tok)

        for tok in self.doc:
            for x in self.use:
                if tok.i == 0:
                    self.feat[tok.i]["prev-1_" + x] = "-BOS-"
                else:
                    self.feat[tok.i]["prev-1_" + x] = self.feat[tok.i - 1][x]
                if (tok.i == 0) or (tok.i == 1):
                    self.feat[tok.i]["prev-2_" + x] = "-BOS-"
                else:
                    self.feat[tok.i]["prev-2_" + x] = self.feat[tok.i - 2][x]
                if tok.i == len(self.sent) - 1:
                    self.feat[tok.i]["next+1_" + x] = "-EOS-"
                else:
                    self.feat[tok.i]["next+1_" + x] = self.feat[tok.i + 1][x]
                if (tok.i == len(self.sent) - 2) or (tok.i == len(self.sent) - 1):
                    self.feat[tok.i]["next+2_" + x] = "-EOS-"
                else:
                    self.feat[tok.i]["next+2_" + x] = self.feat[tok.i + 2][x]

    def is_title(self, tok):
        self.feat[tok.i]["is_title"] = tok.is_title

    def orth_(self, tok):
        self.feat[tok.i]["orth_"] = tok.orth_

    def lemma_(self, tok):
        self.feat[tok.i]["lemma_"] = tok.lemma_

    def lower_(self, tok):
        self.feat[tok.i]["lower_"] = tok.lower_

    def norm_(self, tok):
        self.feat[tok.i]["norm_"] = tok.norm_

    def shape_(self, tok):
        self.feat[tok.i]["shape_"] = tok.shape_

    def prefix_(self, tok):
        self.feat[tok.i]["prefix_"] = tok.prefix_

    def suffix_(self, tok):
        self.feat[tok.i]["suffix_"] = tok.suffix_


def build_features(path):

    print("Generating {0} {1} features...".format(dataset))
    data = []
    sent = []
    feat = {}
    idx = 0

    for cnt, line in enumerate(open(path)):

        if not line.split():
            fb = FeatureBuilder(sent, feat)
            for idx, tok in enumerate(sent):
                feats = fb.feat[idx]
                cols = sorted(feats.keys())
                data.append([feats[x] for x in cols])

            newline = []
            for x in cols:
                newline.append("-NEWLINE-")
            data.append(newline)

            sent = []
            feat = {}
            idx = 0

        else:
            tok, pos, chunk, tag = line.strip().split()
            feat[idx] = {
                "tok": tok,
                "pos": pos,
                "chunk": chunk,
                "tag": tag,
            }
            idx += 1
            sent.append(tok)

    print("Completed Feature generation for {}".format(path))
    df = pd.DataFrame(data, columns=cols)
    return df


def train(df=None):
    features = list(df)

    features.remove("tag")

    vec = DictVectorizer()

    X_train = vec.fit_transform(df[features].to_dict("records"))
    y_train = df["tag"].values
    label_list = list(set(y_train))

    global labels_to_ids, ids_to_labels
    labels_to_ids = {k: v for v, k in enumerate(label_list)}
    ids_to_labels = {v: k for v, k in enumerate(label_list)}
    
    y_train = np.array([labels_to_ids[i] for i in y_train])
    
    print("Training model...")
    print("X", X_train.shape)
    print("y", y_train.shape)

    if model_name == "logreg":
        model = LogisticRegression(multi_class="multinomial", solver="sag", C=2.0,)
    elif model_name == "xgb":
        model = XGBClassifier(booster='gbtree',random_state=42,tree_method='gpu_hist', gpu_id=0,)
    else:
        raise NotImplementedError

    model.fit(X_train, y_train)

    with open(f"{base_path}{dataset}_{MODEL_FP}", "wb") as model_file:
        pickle.dump(model, model_file)

    with open(f"{base_path}{dataset}_{VECTORIZER_FP}", "wb") as vectorizer_file:
        pickle.dump(vec, vectorizer_file)

    print("Model Trained and Saved\n")
    return



def tag(split, df=None):
    features = list(df)
    features.remove("tag")

    model = pickle.load(open(f"{base_path}{dataset}_{MODEL_FP}", "rb"))
    vec = pickle.load(open(f"{base_path}{dataset}_{VECTORIZER_FP}", "rb"))

    X = vec.transform(df[features].to_dict("records"))
    y = df["tag"].values
    y = np.array([labels_to_ids[i] for i in y])

    print("Tagging {0}...".format(split))
    print("X", X.shape)
    print("y", y.shape)

    y_pred = model.predict(X)

    true_labels, preds = [], []
    temp_labels, temp_preds = [], []

    for lab, pre in zip(y, y_pred):
        if lab != "-NEWLINE-":
            temp_labels.append(lab)
            temp_preds.append(pre)
        else:
            true_labels.append(temp_labels)
            preds.append(temp_preds)
    print("\nF1 Score: ", f1_score(true_labels, preds))
    return


def main(train_path, test_path):
    train_df = build_features(train_path)
    test_df = build_features(test_path)
    train(train_df)
    tag("test", test_df)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='NER with word feature engineering')
    parser.add_argument('--base-path', default="./")
    parser.add_argument('--train_path', default="./NER_dataset/train_ncbi.txt")
    parser.add_argument('--test_path', default="./NER_dataset/test_ncbi.txt")
    parser.add_argument('--dataset', default="jnlpba")
    parser.add_argument('--model-name', default="logreg")
    args = parser.parse_args()
    global base_path, dataset, model_name
    base_path = args.base_path
    dataset = args.dataset
    model_name = args.model
    main(args.train_path, args.test_path)
