# ! pip install -q kaggle

# ! mkdir ~/.kaggle

# ! cp kaggle.json ~/.kaggle/
#
# ! chmod 600 ~/.kaggle/kaggle.json
#
# ! kaggle competitions download -c nitro-lang-processing-1
#
# ! mkdir nitro-lang-processing-1
#
# ! unzip nitro-lang-processing-1.zip -d nitro-lang-processing-1
#
# ! pip install unidecode
# ! pip install transformers

# https://huggingface.co/docs/transformers/model_doc/bert
# https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1
# This project is a BERT model based with the pretraining made by Dumitrescu Stefan on a 15GB Romanian corpus
# We fine-tuned a variable number of encoders and the others were frozen

seed = 8

from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForTokenClassification
from tqdm import tqdm

import re
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
np.random.seed(seed)
np.random.RandomState(seed)
from sklearn.utils.class_weight import compute_class_weight

import random
random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

# import jax
# jax.random.PRNGKey(seed)

from unidecode import unidecode

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import json


MAX_LENGTH = 128
EPOCHS = 10
BATCH_SIZE = 16
NUM_LAYERS_FROZEN = 8

with open("./nitro-lang-processing-1/train.json") as fin:
    raw_train_data = json.load(fin)
with open("./nitro-lang-processing-1/test.json") as fin:
    raw_test_data = json.load(fin)
with open("./nitro-lang-processing-1/tag_to_id.json") as fin:
    tag_to_id = json.load(fin)

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
            reshaped_labels.append( item['ner_ids'][i: min(i + reshaped_length, len(item['ner_ids']))])

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

def read_dataset_old(dataset, tokenizer, train=True):
    data = []
    labels = []
    max_length = 0

    for item in dataset:
        prelucrate_item = []
        for token in item['tokens']:
            prelucrate_item.append(re.sub(r"\W+", 'n', token))
        sequence = tokenizer(
            prelucrate_item,
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
            sequence_labels = item['ner_ids']
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

# https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1
# tokenizer is taken from the dumitrescustefan's pretrained BERT module from huggingface
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")

train_data, train_labels = read_dataset(raw_train_data, tokenizer=tokenizer)

X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=seed)

# https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1
# Model is pretrained by Dumitrescu
model = AutoModelForTokenClassification.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1", num_labels=len(tag_to_id))
# for param in model.bert.parameters():
#     param.requires_grad = False

# print(model.bert.encoder.layer)

# Frozen the embedding layer and some of the encoders
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for layer in model.bert.encoder.layer[:NUM_LAYERS_FROZEN]:
    for param in layer.parameters():
        param.requires_grad = False
# ner_model = pipeline('ner', model=model, tokenizer=tokenizer)

test = tokenizer(
    ["testând", "de", "trei", "ori"],
    is_split_into_words=True,
    padding='max_length',
    return_offsets_mapping=True,
    truncation=False,
    max_length=4
)
print(test)
print(tokenizer.convert_ids_to_tokens([4231, 476]))
print(tokenizer.convert_tokens_to_ids(['test', '##ând']))
print(tokenizer.convert_ids_to_tokens([23570]))

print(train_labels[0])

def pad(samples, max_length):

    return torch.tensor([
        sample[:max_length] + [0] * max(0, max_length - len(sample))
        for sample in samples
    ])

# padded_train_data = pad(train_data, 563)
# padded_train_data[0]

# print(padded_train_data.shape)

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


# print(model(torch.tensor(tokenizer.encode("Testing the model", add_special_tokens=True)).unsqueeze(0)))

# print(tokenizer.encode_plus(["Convorbiri", "literare", "."]))
print(tokenizer.convert_tokens_to_ids("[PAD]"))
print(tokenizer.convert_ids_to_tokens(10))

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

# Some ideas were taken from here about the -100 crossEntropyLoss default ignored index so we made the padding/other tokens -100
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb#scrollTo=0jDNXrjr-6BW

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

    return epoch_loss, epoch_acc, labels

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

    return epoch_loss, epoch_acc, labels

weights = np.zeros(16)
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
weights = compute_class_weight(class_weight="balanced", classes=np.arange(0, 16), y=proper_labels)

print(weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# move the model to GPU (when available)
model.to(device)

# create a SGD optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, threshold=1e-2, verbose=True)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
#
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, threshold=1e-3, verbose=True)

# set up loss function
loss_criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float).to(device), ignore_index=-100, reduction="mean")

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
for epoch in range(1, EPOCHS+1):
    train_loss, train_accuracy, pepega_labels = train_epoch(model, train_dataloader, loss_criterion, optimizer, device)
    val_loss, val_accuracy, pepega_pepega_labels = eval_epoch(model, validation_dataloader, loss_criterion, device)
    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    print('\nEpoch %d'%(epoch))
    print('train loss: %10.8f, accuracy: %10.8f'%(train_loss, train_accuracy))
    print('val loss: %10.8f, accuracy: %10.8f'%(val_loss, val_accuracy))

# ! nvidia-smi

model.save_pretrained("model_pretrained_Adam_128_16_god_seed_v4")
torch.save(model.state_dict(), "model_Adam_128_16_clip_god_seed_v4.pt")

# model.load_state_dict(torch.load("model_SGD_64_2.pt"))

# model.from_pretrained("model_pretrained_SGD_64_2")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # move the model to GPU (when available)
# model.to(device)
#
# # create a SGD optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# set up loss function
# loss_criterion = nn.CrossEntropyLoss()
# val_loss, val_accuracy, pepega_pepega_labels = eval_epoch(model, validation_dataloader, loss_criterion, device)
# print(val_accuracy)

def read_test(dataset, tokenizer):
    data = []
    max_length = 0
    counter = 0
    reshaped_dataset = []
    # reshaped_length = 4
    # for item in dataset:
    #     for i in range(0, len(item['tokens']), reshaped_length):
    #         reshaped_dataset.append(item['tokens'][i: min(i + reshaped_length, len(item['tokens']) ) ] )

    for item in dataset:
        reshaped_dataset.append(item['tokens'])

    for item in reshaped_dataset:
        # counter += len(item)
        prelucrate_item = []
        for token in item:
            prelucrate_item.append(re.sub(r"\W+", 'or', token))
        # print(prelucrate_item)
        sequence = tokenizer(
            prelucrate_item,
            is_split_into_words=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            # truncation=True,
            return_offsets_mapping=True
        )
        # sequence = tokenizer.encode(
        #     prelucrate_item,
        #     is_pretokenized=True,
        # )
        sequence = {key: torch.as_tensor(value) for key, value in sequence.items()}
        data.append(sequence)

    #     if len(sequence['input_ids']) > max_length:
    #         print(tokenizer.convert_ids_to_tokens(sequence['input_ids']))
    #         print((sequence['offset_mapping']))
    #         print(item)
    #     max_length = max(len(sequence['input_ids']), max_length)
    # print(max_length)
    return data

reshaped_test_data = read_test(raw_test_data, tokenizer=tokenizer)

test_dataloader = DataLoader(
    dataset=reshaped_test_data,
    batch_size=BATCH_SIZE
)

def test_epoch(model, test_dataloader, device):
    model.eval()
    epoch_loss = 0.0
    num_batches = len(test_dataloader)
    predictions = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_dataloader)):
            batch_data = batch
            sequence_ids = batch_data['input_ids'].to(device)
            sequence_masks = batch_data['attention_mask'].to(device)
            offset_mapping = batch_data['offset_mapping']

            raw_output = model(sequence_ids, attention_mask=sequence_masks)
            output =  raw_output['logits']
            logits = output.view(-1, model.num_labels)
            batch_predictions = torch.argmax(logits, dim=1)
            print(batch_predictions)


            filtered_predictions = []

            # raw_batch_predictions = torch.argmax(output, dim=2)
            # # print(offset_mapping.shape)
            # for index_bt, bt in enumerate(offset_mapping):
            #     for index, offset in enumerate(bt):
            #         if offset[0] == 0 and offset[1] != 0:
            #             filtered_predictions.append(raw_batch_predictions[index_bt][index])

            for index, offset in enumerate(offset_mapping.view(-1, 2)):
                if offset[0] == 0 and offset[1] != 0:
                    filtered_predictions.append(batch_predictions[index])

            predictions += filtered_predictions

    return predictions

all_predictions = test_epoch(model, test_dataloader, device)
print(len(all_predictions))

# print(all_predictions)

g = open("adam_sample_128_16_god_seed_v4.csv", "w")
idx = 0
g.write("Id,ner_label\n")
for pred in all_predictions:
    g.write(str(idx) + "," + str(pred.item()) + "\n")
    idx += 1
g.close()

