import torch
import pandas as pd
import os
import json
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from helpers import nlp_normalization

class SentimentAnalysisDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder, vocab_file, transform=None):
        self.pos_folder = root_folder + "/pos"
        self.neg_folder = root_folder + "/neg"

        # get a list of files in each folder
        self.pos_files = [self.pos_folder + "/" + f for f in os.listdir(self.pos_folder) if f.endswith('.txt')]
        self.pos_labels = [1] * len(self.pos_files)
        self.neg_files = [self.neg_folder + "/" + f for f in os.listdir(self.neg_folder) if f.endswith('.txt')]
        self.neg_labels = [0] * len(self.neg_files)
        self.transform = transform

        # combine the lists for files and labels
        self.files = self.pos_files + self.neg_files
        self.labels = self.pos_labels + self.neg_labels
        
        # Load vocabulary from json file
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
        
        self.word_to_idx = {word: data["index"] for word, data in vocab_data["word_vocab"].items()}
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Get file path and label
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # Read the text file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
        
        # Tokenize the text (simple whitespace tokenization)
        if self.transform is not None:
            tokens = self.transform(text)
        else:
            tokens = text.lower().split()

        # Convert tokens to indices using vocabulary
        token_indices = []
        for token in tokens:
            if token in self.word_to_idx:
                token_indices.append(self.word_to_idx[token])
            else:
                # Use <UNK> token for unknown words
                token_indices.append(self.word_to_idx["<UNK>"])
        # Convert list of indices to tensor
        features = torch.tensor(token_indices, dtype=torch.long)
        
        # Convert to tensor
        label = torch.tensor(label, dtype=torch.long)
        
        return features, label

def build_dataloader(root_folder, vocab_file, batch_size, shuffle=True,
                     split=False, val_ratio=0.1, random_seed=42):

    
    # Load vocabulary from json file
    with open(vocab_file, 'r') as f:
        vocab_data = json.load(f)

    nlp_transform=partial(nlp_normalization.nlp_transform, 
                      normalization=vocab_data['config']["normalization"], 
                      remove_stopwords=vocab_data['config']["remove_stopwords"])

    dataset = SentimentAnalysisDataset(root_folder, vocab_file, nlp_transform)

    def collate_fn(batch):
        # Separate features and labels
        features = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Pad sequences to the maximum length in this batch
        features_padded = pad_sequence(features, batch_first=True, padding_value=0)

        # Convert labels into a tensor
        labels_tensor = torch.stack(labels)

        # Store original sequence lengths
        lengths = torch.tensor([len(feature) for feature in features], dtype=torch.long)

        return features_padded, labels_tensor, lengths
    
    # Split dataset if required
    if split:
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        
        # Calculate sizes for training and validation splits
        dataset_size = len(dataset)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size]
        )
        
        # Create separate DataLoaders for training and validation
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # No need to shuffle validation data
            collate_fn=collate_fn
        )

        print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
        return train_loader, val_loader
    
    else:
        # Return a single DataLoader for the entire dataset
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn
        )



