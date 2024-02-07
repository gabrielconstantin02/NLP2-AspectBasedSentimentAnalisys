from tqdm import tqdm
import os
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig, AutoModelForTokenClassification
from sklearn.metrics import f1_score, classification_report
import numpy as np

from utils.dataset import ABSADataset
from utils.general import freeze_model, calculate_weights, plot_basic, get_next_name, save_setup, compute_metrics_absa
from models.heads import BertABSATagger

from utils.config import CFG

os.environ["TOKENIZERS_PARALLELISM"] = "false"
np.random.seed(CFG.SEED)
np.random.RandomState(CFG.SEED)
random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)


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

        batch_acc = compute_metrics_absa(filtered_predictions.cpu().numpy(), filtered_labels.cpu().numpy())['micro-f1']
        epoch_acc += batch_acc

        loss_scalar = loss.item()

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
            loss = loss_crt(logits, batch_labels.view(-1))

            filtered_labels = torch.masked_select(batch_labels.view(-1), proper_labels)
            filtered_predictions = torch.masked_select(batch_predictions, proper_labels)

            labels += filtered_labels.squeeze().tolist()
            predictions += filtered_predictions.tolist()

            batch_acc = compute_metrics_absa(filtered_predictions.cpu().numpy(), filtered_labels.cpu().numpy())['micro-f1']
            epoch_acc += batch_acc

            loss_scalar = loss.item()

            epoch_loss += loss_scalar

    epoch_loss = epoch_loss/num_batches
    epoch_acc = epoch_acc/num_batches

    return epoch_loss, epoch_acc, predictions, labels

def train(model, train_dataloader, val_dataloader, weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move the model to GPU (when available)
    model.to(device)

    # create a AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, threshold=1e-2, verbose=True)


    # set up loss function
    loss_criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float).to(device), ignore_index=-100, reduction="mean")

    best_val_acc = 0.0
    best_model = model
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_pred, val_labels = None, None
    val_accuracies = []
    for epoch in range(1, CFG.EPOCHS+1):
        print('\nEpoch %d'%(epoch))

        train_loss, train_accuracy = train_epoch(model, train_dataloader, loss_criterion, optimizer, device)
        val_loss, val_accuracy, val_pred_ep, val_labels_ep = eval_epoch(model, val_dataloader, loss_criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print('train loss: %10.8f, accuracy: %10.8f'%(train_loss, train_accuracy))
        print('val loss: %10.8f, accuracy: %10.8f'%(val_loss, val_accuracy))
        print("Classification report:")
        print(classification_report(val_labels_ep, val_pred_ep))
        print("ABSA eval report:")
        print(compute_metrics_absa(val_pred_ep, val_labels_ep))

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model = model
            torch.save(model.state_dict(), os.path.join(CFG.SAVE_PATH, f'model_ep_{epoch}_{val_accuracy}.pt'))
            val_pred, val_labels = val_pred_ep, val_labels_ep

    plot_basic(train_losses, train_accuracies, val_losses, val_accuracies, val_pred, val_labels)

def get_cmd_args():
    global CFG
    parser = argparse.ArgumentParser(description='Python script that trains a CNN model on generated images.')
    parser.add_argument('--model_name', type=str, help='Model to be used', default="BERT", \
                        choices=['BERT', "ABSA_BERT"])
    parser.add_argument('--absa_type', type=str, help='Type of head to be used', default="linear", \
                        choices=['linear', "lstm", 'gru', 'tfm', 'san', 'crf'])
    parser.add_argument('--fix_tfm', type=str, help="Whether to freeze the whole model or partial or none", default="partial", \
                        choices=['full', "partial", "none"])
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train', default="20")
    parser.add_argument('--batch_size', type=int, help='Batch size', default="16")
    parser.add_argument('--data_path', type=str, help="The dataset path", default="data/laptop14", \
                        choices=['data/laptop14', "data/rest"])
    args = parser.parse_args()

    CFG.EPOCHS = args.num_epochs
    CFG.DATA_PATH = args.data_path
    CFG.BATCH_SIZE = args.batch_size
    return args

if __name__ == "__main__":
    args = get_cmd_args()

    CFG.SAVE_PATH = get_next_name(CFG.SAVE_PATH)
    os.makedirs(CFG.SAVE_PATH, exist_ok=True) 

    save_setup(CFG.SAVE_PATH, CFG.SAVE_FILES, args)
    # prepare the data
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = ABSADataset(os.path.join(CFG.DATA_PATH, 'train.txt'), tokenizer, max_length=CFG.MAX_LENGTH)
    validation_dataset = ABSADataset(os.path.join(CFG.DATA_PATH, 'test.txt'), tokenizer, max_length=CFG.MAX_LENGTH)

    dataset_weights = calculate_weights(train_dataset.labels, 'sklearn')

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True
    )
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=CFG.BATCH_SIZE
    )
    # prepare the model
    if args.model_name == "BERT":
        model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=4) # {'O': 0, 'T-POS': 1, 'T-NEG': 2, 'T-NEU': 3}
        freeze_model(model, CFG.NUM_LAYERS_FROZEN)
    else:
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=4)
        config.absa_type = args.absa_type
        config.fix_tfm = args.fix_tfm
        model = BertABSATagger('bert-base-uncased', config)
        # fix the parameters in BERT and regard it as feature extractor
        if config.fix_tfm == "full":
            freeze_model(model, config.num_hidden_layers)
        elif config.fix_tfm == "partial":
            freeze_model(model, CFG.NUM_LAYERS_FROZEN)


    train(model, train_dataloader, validation_dataloader, dataset_weights)