from utils import get_mask
import torch
from torch.nn.utils import clip_grad_norm_
from seqeval.metrics import f1_score


def train_model(dataloader, model, optimizer, batch_num, use_gpu=False):
    model.train()
    for batch in dataloader:
        batch_num += 1
        model.zero_grad()
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']
        if use_gpu:
            batch_text = batch_text.cuda()
            batch_label = batch_label.cuda()
        mask = get_mask(batch_text)
        loss = model.neg_log_likelihood_loss(batch_text, seq_length, batch_label, mask)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return batch_num


def evaluate(dataloader, model, label_vocab, use_gpu=False):
    model.eval()
    true_lables, preds = [], []
    for batch in dataloader:
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']
        if use_gpu:
            batch_text = batch_text.cuda()
            batch_label = batch_label.cuda()
        mask = get_mask(batch_text)
        with torch.no_grad():
            tag_seq = model(batch_text, seq_length, batch_label, mask)

        for line_tesor, labels_tensor, predicts_tensor in zip(batch_text, batch_label, tag_seq):
            temp1, temp2 = [], []
            for word_tensor, label_tensor, predict_tensor in zip(line_tesor, labels_tensor, predicts_tensor):
                if word_tensor.item() == 0:
                    break
                temp1.append(label_vocab.id_to_label(label_tensor.item()))
                temp2.append(label_vocab.id_to_label(predict_tensor.item()))
            true_lables.append(temp1)
            preds.append(temp2)

    new_f1 = f1_score(true_lables, preds)
    return new_f1
