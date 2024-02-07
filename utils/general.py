import re
import os
import json
import shutil
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

from utils.config import CFG

def plot_extra(accuracies_macro, accuracies_micro):
    fig, axis = plt.subplots(1,2, figsize=(10,5))
    epochs_list = range(CFG.EPOCHS)

    # creating the 4 plots
    axis[0].plot(epochs_list, accuracies_macro)
    axis[0].set_title("Val macro")
    axis[1].plot(epochs_list, accuracies_micro)
    axis[1].set_title("Val micro")

    # asigning the label
    for ax in axis.flat:
        ax.set(xlabel='Epochs', ylabel='Accuracy')

    # making the axis show only outer label
    # for ax in axis.flat:
    #     ax.label_outer()

    # saving the figure
    plt.savefig(os.path.join(CFG.SAVE_PATH, "model_plots_extra.png"))
    plt.close()


def plot_basic(train_losses, train_accuracies, val_losses, val_accuracies, val_pred_ep, val_labels_ep):
    fig, axis = plt.subplots(2,2, figsize=(10,10))
    epochs_list = range(CFG.EPOCHS)

    # creating the 4 plots
    axis[0,0].plot(epochs_list, train_losses)
    axis[0,0].set_title("Train losses")
    axis[0,1].plot(epochs_list, val_losses)
    axis[0,1].set_title("Val losses")
    axis[1,0].plot(epochs_list, train_accuracies)
    axis[1,0].set_title("Train accuracy")
    axis[1,1].plot(epochs_list, val_accuracies)
    axis[1,1].set_title("Val accuracy")

    # asigning the label
    for ax in axis.flat:
        ax.set(xlabel='Epochs', ylabel='Loss')

    axis.flat[2].set(xlabel='Epochs', ylabel="Accuracy")

    # making the axis show only outer label
    for ax in axis.flat:
        ax.label_outer()

    # saving the figure
    plt.savefig(os.path.join(CFG.SAVE_PATH, "model_plots.png"))
    plt.close()

    ###### plotting the confusion matrix
    # creating the confusion matrix
    val_confusion_matrix = confusion_matrix(val_labels_ep, val_pred_ep)
    val_confusion_matrix_df = pd.DataFrame(val_confusion_matrix, range(CFG.NUM_LABELS), range(CFG.NUM_LABELS))
    # settings for the plot
    sns.set(font_scale=0.8)
    sns.set(rc={'figure.figsize':(18,16)})
    # plot the confusion matrix
    sns.heatmap(val_confusion_matrix_df, annot=True, annot_kws={"size":12}, norm=LogNorm(), fmt='d')
    # sns.heatmap(val_confusion_matrix_df, cbar=True, annot=False, fmt='d')

    plt.savefig(os.path.join(CFG.SAVE_PATH, "model_confusion_matrix.png"))
    plt.close()

def parse_input_file(filepaths):
    label_mapper = CFG.LABELS_MAPPING
    if os.path.exists(filepaths):
        filepaths = [filepaths]
    else:
        filepath = filepaths.split('/')
        keyword = filepath[-2]
        file_type = filepath[-1]
        filepath = '/'.join(filepath[:-2])
        filepaths = []
        for path in os.listdir(filepath):
            if keyword in path.split('/')[-1]:
                filepaths.append(os.path.join(os.path.join(filepath, path), file_type))
    
    dataset = []
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                item = {
                    'tokens': [],
                    'labels': []
                }
                _, token_label_pairs = line.strip().split("####")
                for token_label in token_label_pairs.split(" "):
                    # print(token_label)
                    token_label_split = token_label.split("=")
                    if len(token_label_split) == 2:
                        token, label = token_label.split("=")
                    else:
                        token = ''.join((len(token_label) - 2) * ['='])
                        label = token_label[-1]
                    item['tokens'].extend([token])
                    item['labels'].extend([label_mapper[label]])
                dataset.append(item)

    return dataset
    

def fix_max_length(dataset, reshaped_length=110):
    reshaped_dataset = []
    reshaped_labels = []
    for item in dataset:
        prelucrate_item = []
        for token in item['tokens']:
            prelucrate_item.append(re.sub(r"\W+", 'n', token))
        for i in range(0, len(prelucrate_item), reshaped_length):
            reshaped_dataset.append(prelucrate_item[i: min(i + reshaped_length, len(prelucrate_item))])
            reshaped_labels.append( item['labels'][i: min(i + reshaped_length, len(item['labels']))])
    return reshaped_dataset, reshaped_labels


def freeze_model(model, num_layers):
    # Frozen the embedding layer and some of the encoders
    for param in model.bert.embeddings.parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[:num_layers]:
        for param in layer.parameters():
            param.requires_grad = False


def pad(samples, max_length):
    return torch.tensor([
        sample[:max_length] + [0] * max(0, max_length - len(sample))
        for sample in samples
    ])


def calculate_weights(labels, mode='sklearn'):
    weights = np.zeros(4)
    proper_labels = []
    for sequence in labels:
        for label in sequence:
            if label != -100:
                proper_labels.append(int(label))
                weights[label] += 1
    if mode == 'sklearn':
        weights = compute_class_weight(class_weight="balanced", classes=np.arange(0, 4), y=proper_labels)
    else: 
        max_weight = np.max(weights)
        for index in range(len(weights)):
            weights[index] = max_weight / weights[index]
    return weights

def get_next_name(path):
    """
    This gets the next path to save a train as.
    It is made in a incremental way.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    all_paths = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("train")]

    all_paths_numbers = [int(train[len("train"):]) for train in all_paths]
    if len(all_paths_numbers) == 0:
        next_number = 1
    else:
        next_number = max(all_paths_numbers) + 1

    new_path = os.path.join(path, f'train{next_number:03d}')

    return new_path


def save_setup(path, files, args):
    """
    This function saves the current setup of the train in the train subfolder
    Main purpose is full reproductibility of each model
    """
    final_path = os.path.join(path, "setup")
    try:
        os.mkdir(final_path)
    except:
        raise Exception(f"The following path shouldn't exist: {final_path}")
    with open(os.path.join(final_path, "parse_args.json"), 'w') as file:
        json.dump(vars(args), file, indent=4)
    for file in files:
        file_path = os.path.join('./', file)
        dest_path = os.path.join(final_path, file)
        if os.path.isdir(file_path):
            shutil.copytree(file_path, dest_path)
        else:
            shutil.copy2(file_path, dest_path)
    
# TODO: translate this to work with our setup
def ot2bieos_ts(ts_tag_sequence):
    """
    ot2bieos function for targeted-sentiment task, ts refers to targeted -sentiment / aspect-based sentiment
    :param ts_tag_sequence: tag sequence for targeted sentiment
    :return:
    """
    n_tags = len(ts_tag_sequence)
    new_ts_sequence = []
    prev_pos = '$$$'
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O' or cur_ts_tag == 'EQ':
            # when meet the EQ label, regard it as O label
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            # cur_pos is T
            if cur_pos != prev_pos:
                # prev_pos is O and new_cur_pos can only be B or S
                if i == n_tags - 1:
                    new_ts_sequence.append('S-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('S-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('B-%s' % cur_sentiment)
            else:
                # prev_pos is T and new_cur_pos can only be I or E
                if i == n_tags - 1:
                    new_ts_sequence.append('E-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence

def tag2ts(ts_tag_sequence):
    """
    transform ts tag sequence to targeted sentiment
    :param ts_tag_sequence: tag sequence for ts task
    :return:
    """
    n_tags = len(ts_tag_sequence)
    ts_sequence, sentiments = [], []
    beg, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        # current position and sentiment
        # tag O and tag EQ will not be counted
        eles = ts_tag.split('-')
        if len(eles) == 2:
            pos, sentiment = eles
        else:
            pos, sentiment = 'O', 'O'
        if sentiment != 'O':
            # current word is a subjective word
            sentiments.append(sentiment)
        if pos == 'S':
            # singleton
            ts_sequence.append((i, i, sentiment))
            sentiments = []
        elif pos == 'B':
            beg = i
            if len(sentiments) > 1:
                # remove the effect of the noisy I-{POS,NEG,NEU}
                sentiments = [sentiments[-1]]
        elif pos == 'E':
            end = i
            # schema1: only the consistent sentiment tags are accepted
            # that is, all of the sentiment tags are the same
            if end > beg > -1 and len(set(sentiments)) == 1:
                ts_sequence.append((beg, end, sentiment))
                sentiments = []
                beg, end = -1, -1
    return ts_sequence

def match_ts(gold_ts_sequence, pred_ts_sequence):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    # positive, negative and neutral
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_ts_sequence:
        #print(t)
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count

def compute_metrics_absa(preds, labels):
    absa_label_vocab = CFG.LABELS_MAPPING
    absa_id2tag = {}
    for k in absa_label_vocab:
        v = absa_label_vocab[k]
        absa_id2tag[v] = k
    # number of true postive, gold standard, predicted targeted sentiment
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
    # precision, recall and f1 for aspect-based sentiment analysis
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)
    class_count = np.zeros(3)
    pred_labels = preds
    gold_labels = labels
    assert len(pred_labels) == len(gold_labels)
    # here, no EQ tag will be induced
    pred_tags = [absa_id2tag[label] for label in pred_labels]
    gold_tags = [absa_id2tag[label] for label in gold_labels]

    gold_tags = ot2bieos_ts(gold_tags)
    pred_tags = ot2bieos_ts(pred_tags)
    g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=gold_tags), tag2ts(ts_tag_sequence=pred_tags)

    hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                            pred_ts_sequence=p_ts_sequence)
    n_tp_ts += hit_ts_count
    n_gold_ts += gold_ts_count
    n_pred_ts += pred_ts_count
    for (b, e, s) in g_ts_sequence:
        if s == 'POS':
            class_count[0] += 1
        if s == 'NEG':
            class_count[1] += 1
        if s == 'NEU':
            class_count[2] += 1
    for i in range(3):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + CFG.SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + CFG.SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + CFG.SMALL_POSITIVE_CONST)

    macro_f1 = ts_f1.mean()

    # calculate micro-average scores for ts task
    # TP
    n_tp_total = sum(n_tp_ts)
    # TP + FN
    n_g_total = sum(n_gold_ts)
    # print("class_count:", class_count)

    # TP + FP
    n_p_total = sum(n_pred_ts)
    micro_p = float(n_tp_total) / (n_p_total + CFG.SMALL_POSITIVE_CONST)
    micro_r = float(n_tp_total) / (n_g_total + CFG.SMALL_POSITIVE_CONST)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + CFG.SMALL_POSITIVE_CONST)
    scores = {'macro-f1': macro_f1, 'precision': micro_p, "recall": micro_r, "micro-f1": micro_f1, 'class_count': class_count}
    return scores