seed = 8
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForTokenClassification
from tqdm import tqdm

import re
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
np.random.seed(seed)
np.random.RandomState(seed)
from sklearn.utils.class_weight import compute_class_weight

import random
random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader



MAX_LENGTH = 128
EPOCHS = 10
BATCH_SIZE = 16
NUM_LAYERS_FROZEN = 8
MODEL_NAME = "laptop_bert_uncased_v4"

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
    

def read_dataset(dataset, tokenizer, train=True):
    data = []
    labels = []
    max_length = 0
    reshaped_dataset = []
    reshaped_labels = []
    reshaped_length = 110
    for item in dataset:
        prelucrate_item = []
        for token in item['tokens']:
            prelucrate_item.append(re.sub(r"\W+", 'n', token))
        for i in range(0, len(prelucrate_item), reshaped_length):
            reshaped_dataset.append(prelucrate_item[i: min(i + reshaped_length, len(prelucrate_item))])
            reshaped_labels.append( item['labels'][i: min(i + reshaped_length, len(item['labels']))])

    for index in range(len(reshaped_dataset)):
        items, sequence_labels =  reshaped_dataset[index], reshaped_labels[index]
        sequence = tokenizer(
            items,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True

        )
        sequence = {key: torch.as_tensor(value) for key, value in sequence.items()}
        data.append(sequence)

        if train:
            encoded_labels = np.ones(len(sequence["offset_mapping"]), dtype=int) * -100
            # set only labels whose first offset position is 0 and the second is not 0
            i = 0
            for idx, offsets in enumerate(sequence["offset_mapping"]):
                if offsets[0] == 0 and offsets[1] != 0:
                    # overwrite label
                    encoded_labels[idx] = sequence_labels[i]
                    i += 1

            # max_length = max(len(sequence), max_length)
            labels.append(torch.as_tensor(encoded_labels))
    # print(max_length)
    if train:
        return data, labels

    return data


raw_train_data = parse_input_file('data/laptop14/train.txt')
raw_valid_data = parse_input_file('data/laptop14/test.txt')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

X_train, y_train = read_dataset(raw_train_data, tokenizer=tokenizer)
X_val, y_val = read_dataset(raw_valid_data, tokenizer=tokenizer)


model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=4) # {'O': 0, 'T-POS': 1, 'T-NEG': 2, 'T-NEU': 3}
# for param in model.bert.parameters():
#     param.requires_grad = False

# print(model.bert.encoder.layer)

# Frozen the embedding layer and some of the encoders
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for layer in model.bert.encoder.layer[:NUM_LAYERS_FROZEN]:
    for param in layer.parameters():
        param.requires_grad = False


def pad(samples, max_length):

    return torch.tensor([
        sample[:max_length] + [0] * max(0, max_length - len(sample))
        for sample in samples
    ])


class MyDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

class TestDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


train_dataset = MyDataset(X_train, y_train)
validation_dataset = MyDataset(X_val, y_val)

print(len(X_train))
print(len(y_train))

# instantiate the DataLoaders
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
validation_dataloader = DataLoader(
    dataset=validation_dataset,
    batch_size=BATCH_SIZE
)

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

        if idx % 500 == 0:
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

weights = np.zeros(4)
proper_labels = []
for sequence in y_train:
    mini_label = []
    for label in sequence:
        if label != -100:
            # mini_label.append(label)
            proper_labels.append(int(label))
            weights[label] += 1
    # proper_labels.append(mini_label)
max_weight = np.max(weights)
for index, weight in enumerate(weights):
    weights[index] = max_weight / weights[index]
print(weights)
weights = compute_class_weight(class_weight="balanced", classes=np.arange(0, 4), y=proper_labels)

print(weights)

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
    val_loss, val_accuracy = eval_epoch(model, validation_dataloader, loss_criterion, device)
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    print('train loss: %10.8f, accuracy: %10.8f'%(train_loss, train_accuracy))
    print('val loss: %10.8f, accuracy: %10.8f'%(val_loss, val_accuracy))


model.save_pretrained(MODEL_NAME)
torch.save(model.state_dict(), MODEL_NAME + ".pt")
