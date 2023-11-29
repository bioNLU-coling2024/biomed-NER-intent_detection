import random
import torch
import numpy as np
import argparse
import os
from utils import WordVocabulary, LabelVocabulary, my_collate_fn, lr_decay
from dataset import MyDataset
from torch.utils.data import DataLoader
from model import NamedEntityRecog
import torch.optim as optim
from train import train_model, evaluate

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def main(args):
    use_gpu = torch.cuda.is_available()
    print(args, "\n\n")

    word_vocab = WordVocabulary(args.train_path, args.test_path, args.number_normalized)
    label_vocab = LabelVocabulary(args.train_path)

    train_dataset = MyDataset(args.train_path, word_vocab, label_vocab, args.number_normalized)
    test_dataset = MyDataset(args.test_path, word_vocab, label_vocab, args.number_normalized)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    model = NamedEntityRecog(word_vocab.size(), args.word_embed_dim, args.word_hidden_dim,
                            args.feature_extractor, label_vocab.size(), args.dropout, use_crf=args.use_crf,use_gpu=use_gpu)

    if use_gpu:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    print("Training begin\n\n")

    batch_num = -1
    best_f1 = -1
    early_stop = 0

    for epoch in range(args.epochs):
        print('train {}/{} epoch'.format(epoch + 1, args.epochs))
        optimizer = lr_decay(optimizer, epoch, 0.05, args.lr)
        batch_num = train_model(train_dataloader, model, optimizer, batch_num, use_gpu)
        new_f1 = evaluate(test_dataloader, model, label_vocab, use_gpu)
        print('f1 is {0:.2f} at {1}th epoch on dev set'.format(new_f1, epoch + 1))
        if new_f1 > best_f1:
            best_f1 = new_f1
            print('new best f1 on dev set: {0:.2f}'.format(best_f1))
            early_stop = 0
        else:
            early_stop += 1

        print('train {}th epoch '.format(epoch + 1))
        print()

        if early_stop > args.patience:
            print('early stop')
            break
    print('\n\nTraining end')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Named Entity Recognition Model')
    parser.add_argument('--word_embed_dim', type=int, default=100)
    parser.add_argument('--word_hidden_dim', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--feature_extractor', choices=['lstm', 'cnn'], default='cnn')
    parser.add_argument('--use_crf', type=bool, default=True)
    parser.add_argument('--train_path', default="../NER_dataset/train_ncbi.txt")
    parser.add_argument('--test_path', default="../NER_dataset/test_ncbi.txt")
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--number_normalized', type=bool, default=True)

    args = parser.parse_args()
    main(args)
    
