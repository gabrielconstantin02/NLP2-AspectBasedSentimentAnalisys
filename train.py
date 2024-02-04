from tqdm import tqdm
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForTokenClassification
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

from utils.dataset import ABSADataset
from utils.general import freeze_model, calculate_weights

seed = 8
MAX_LENGTH = 128
EPOCHS = 10
BATCH_SIZE = 16
NUM_LAYERS_FROZEN = 8
MODEL_NAME = "laptop_bert_uncased_v4"
DATA_PATH = 'data/laptop14'

np.random.seed(seed)
np.random.RandomState(seed)
random.seed(seed)
torch.manual_seed(seed)


def train_epoch(model, train_dataloader, loss_crt, optimizer, device):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = len(train_dataloader)
    predictions = []
    labels = []
    for idx, batch in tqdm(enumerate(train_dataloader)):
        batch_data, batch_labels = batch
        sequence_ids = batch_data['input_ids'].to(device, dtype=torch.long)
        sequence_masks = batch_data['attention_mask'].to(device)
        batch_labels = batch_labels.to(device)

        raw_output = model(input_ids=sequence_ids, attention_mask=sequence_masks, labels=batch_labels)
        loss, output = raw_output['loss'], raw_output['logits']
        logits = output.view(-1, model.num_labels)
        batch_predictions = torch.argmax(logits, dim=1)

        proper_labels = batch_labels.view(-1) != -100
        loss = loss_crt(logits, batch_labels.view(-1))

        filtered_labels = torch.masked_select(batch_labels.view(-1), proper_labels)
        filtered_predictions = torch.masked_select(batch_predictions, proper_labels)

        labels += filtered_labels.squeeze().tolist()
        predictions += filtered_predictions.tolist()

        batch_acc = accuracy_score(filtered_labels.cpu().numpy(), filtered_predictions.cpu().numpy())
        epoch_acc += batch_acc

        loss_scalar = loss.item()

        if idx % 100 == 0:
            print(epoch_acc/(idx + 1))
            print(batch_predictions)

        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=10
        )

        model.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss_scalar

    epoch_loss = epoch_loss/num_batches
    epoch_acc = epoch_acc/num_batches

    return epoch_loss, epoch_acc


def eval_epoch(model, val_dataloader, loss_crt, device):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    num_batches = len(val_dataloader)
    predictions = []
    labels = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_dataloader)):
            batch_data, batch_labels = batch
            sequence_ids = batch_data['input_ids'].to(device, dtype=torch.long)
            sequence_masks = batch_data['attention_mask'].to(device)
            batch_labels = batch_labels.to(device)

            raw_output = model(input_ids=sequence_ids, attention_mask=sequence_masks, labels=batch_labels)
            loss, output = raw_output['loss'], raw_output['logits']
            logits = output.view(-1, model.num_labels)
            batch_predictions = torch.argmax(logits, dim=1)

            proper_labels = batch_labels.view(-1) != -100

            filtered_labels = torch.masked_select(batch_labels.view(-1), proper_labels)
            filtered_predictions = torch.masked_select(batch_predictions, proper_labels)

            labels += filtered_labels.squeeze().tolist()
            predictions += filtered_predictions.tolist()

            batch_acc = accuracy_score(filtered_labels.cpu().numpy(), filtered_predictions.cpu().numpy())
            epoch_acc += batch_acc

            loss_scalar = loss.item()

            epoch_loss += loss_scalar

    epoch_loss = epoch_loss/num_batches
    epoch_acc = epoch_acc/num_batches

    print(classification_report(labels, predictions))
    return epoch_loss, epoch_acc

def train(model, train_dataloader, val_dataloader, weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move the model to GPU (when available)
    model.to(device)

    # create a AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, threshold=1e-2, verbose=True)

    # set up loss function
    loss_criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float).to(device), ignore_index=-100, reduction="mean")

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(1, EPOCHS+1):
        print('\nEpoch %d'%(epoch))
        train_loss, train_accuracy = train_epoch(model, train_dataloader, loss_criterion, optimizer, device)
        val_loss, val_accuracy = eval_epoch(model, val_dataloader, loss_criterion, device)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print('train loss: %10.8f, accuracy: %10.8f'%(train_loss, train_accuracy))
        print('val loss: %10.8f, accuracy: %10.8f'%(val_loss, val_accuracy))


    model.save_pretrained(MODEL_NAME)
    torch.save(model.state_dict(), MODEL_NAME + ".pt")

if __name__ == "__main__":
    # prepare the data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = ABSADataset(os.path.join(DATA_PATH, 'train.txt'), tokenizer, max_length=MAX_LENGTH)
    validation_dataset = ABSADataset(os.path.join(DATA_PATH, 'test.txt'), tokenizer, max_length=MAX_LENGTH)

    dataset_weights = calculate_weights(train_dataset.labels, 'sklearn')

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE
    )
    # prepare the model
    model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=4) # {'O': 0, 'T-POS': 1, 'T-NEG': 2, 'T-NEU': 3}
    freeze_model(model, NUM_LAYERS_FROZEN)

    train(model, train_dataloader, validation_dataloader, dataset_weights)