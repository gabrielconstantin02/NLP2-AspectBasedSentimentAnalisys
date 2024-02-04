import re
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

from utils.config import CFG

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
    val_confusion_matrix_df = pd.DataFrame(val_confusion_matrix, range(CFG.NUM_LABELS), range(CFG.NUM))
    # settings for the plot
    sns.set(font_scale=0.8)
    sns.set(rc={'figure.figsize':(18,16)})
    # plot the confusion matrix
    sns.heatmap(val_confusion_matrix_df, annot=True, annot_kws={"size":12}, norm=LogNorm(), fmt='d')
    # sns.heatmap(val_confusion_matrix_df, cbar=True, annot=False, fmt='d')

    plt.savefig(os.path.join(CFG.SAVE_PATH, "model_confusion_matrix.png"))
    plt.close()

def parse_input_file(filepath):
    label_mapper = {'O': 0, 'T-POS': 1, 'T-NEG': 2, 'T-NEU': 3}
    with open(filepath) as f:
        dataset = []
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