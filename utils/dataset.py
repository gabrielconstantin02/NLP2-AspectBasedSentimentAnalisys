from torch.utils.data import Dataset
from utils.general import parse_input_file, fix_max_length
import torch
import numpy as np

class ABSADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        super().__init__()
        dataset = parse_input_file(data_path)
        data = []
        labels = []
        reshaped_dataset, reshaped_labels = fix_max_length(dataset)

        for index in range(len(reshaped_dataset)):
            items, sequence_labels =  reshaped_dataset[index], reshaped_labels[index]
            sequence = tokenizer(
                items,
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True
            )
            sequence = {key: torch.as_tensor(value) for key, value in sequence.items()}
            data.append(sequence)

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

        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
