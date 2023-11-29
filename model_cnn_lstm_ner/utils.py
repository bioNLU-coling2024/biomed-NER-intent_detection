import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


class WordVocabulary(object):
    def __init__(self, train_path, test_path, number_normalized):
        self.number_normalized = number_normalized
        self._id_to_word = []
        self._word_to_id = {}
        self._pad = -1
        self._unk = -1
        self.index = 0

        self._id_to_word.append('<PAD>')
        self._word_to_id['<PAD>'] = self.index
        self._pad = self.index
        self.index += 1
        self._id_to_word.append('<UNK>')
        self._word_to_id['<UNK>'] = self.index
        self._unk = self.index
        self.index += 1

        with open(train_path, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    if word not in self._word_to_id:
                        self._id_to_word.append(word)
                        self._word_to_id[word] = self.index
                        self.index += 1

        with open(test_path, 'r', encoding='utf-8') as f3:
            lines = f3.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    if word not in self._word_to_id:
                        self._id_to_word.append(word)
                        self._word_to_id[word] = self.index
                        self.index += 1

    def unk(self):
        return self._unk

    def pad(self):
        return self._pad

    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk()

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def items(self):
        return self._word_to_id.items()


class LabelVocabulary(object):
    def __init__(self, filename):
        self._id_to_label = []
        self._label_to_id = {}
        self._pad = -1
        self.index = 0

        self._id_to_label.append('<PAD>')
        self._label_to_id['<PAD>'] = self.index
        self._pad = self.index
        self.index += 1

        with open(filename, 'r', encoding='utf-8') as f1:
            lines = f1.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    label = pairs[-1]

                    if label not in self._label_to_id:
                        self._id_to_label.append(label)
                        self._label_to_id[label] = self.index
                        self.index += 1

    def pad(self):
        return self._pad

    def size(self):
        return len(self._id_to_label)

    def label_to_id(self, label):
        return self._label_to_id[label]

    def id_to_label(self, cur_id):
        return self._id_to_label[cur_id]

def my_collate(batch_tensor):
    word_seq_lengths = torch.LongTensor(list(map(len, batch_tensor)))
    _, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    batch_tensor.sort(key=lambda x: len(x), reverse=True)
    tensor_length = [len(sq) for sq in batch_tensor]
    batch_tensor = pad_sequence(batch_tensor, batch_first=True, padding_value=0)
    return batch_tensor, tensor_length, word_perm_idx


def my_collate_fn(batch):
    return {key: my_collate([d[key] for d in batch]) for key in batch[0]}


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr / (1 + decay_rate * epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def get_mask(batch_tensor):
    mask = batch_tensor.eq(0)
    mask = mask.eq(0)
    return mask
