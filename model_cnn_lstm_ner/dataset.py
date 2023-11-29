from torch.utils.data import Dataset
from utils import normalize_word
import torch
import random


class MyDataset(Dataset):
    def __init__(self, file_path, word_vocab, label_vocab, number_normalized, train_size=1):
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.number_normalized = number_normalized
        texts, labels = [], []
        text, label = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    text.append(word)
                    label.append(pairs[-1])

                else:
                    if len(text) > 0:
                        texts.append(text)
                        labels.append(label)

                    text, label = [], []

        full_data = [(i,j) for i,j in zip(texts, labels)]
        full_data = random.sample(full_data, len(full_data)*train_size)
        self.texts = [i for i,_ in full_data]
        self.labels = [j for _,j in full_data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        text_id = []
        label_id = []
        text = self.texts[item]
        label = self.labels[item]

        for word in text:
            text_id.append(self.word_vocab.word_to_id(word))
        text_tensor = torch.tensor(text_id).long()
        for label_ele in label:
            label_id.append(self.label_vocab.label_to_id(label_ele))
        label_tensor = torch.tensor(label_id).long()

        return {'text': text_tensor, 'label': label_tensor}
