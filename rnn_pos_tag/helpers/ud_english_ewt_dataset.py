import torch
import pandas as pd


class POSTaggingDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file):
        self.df = pd.read_csv(tsv_file, sep='\t')
        data = []  # Use a local list to collect your data

        for _, row in self.df.iterrows():
            words = row['word_indices'].split()
            tags = row['tag_indices'].split()
            data.append(([int(it) for it in words], [int(it) for it in tags]))
        
        # Convert to tensor once after the loop, if desired
        self.data = data  # or process further as needed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        word_indices, tag_indices = self.data[idx]
        return torch.LongTensor(word_indices), torch.LongTensor(tag_indices)


def build_dataloader(tsv_file, batch_size=32, shuffle=True, 
                     split=True, val_ratio=0.1, random_seed=42):
    """
    Create a PyTorch DataLoader for POS tagging data.
    
    Args:
        tsv_file (str): Path to the tsv file containing sentences and tags
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the dataset
        split (bool): Whether to split the dataset into training and validation sets
        val_ratio (float): Ratio of validation data (default: 0.1 or 10%)
        random_seed (int): Random seed for reproducibility

    Returns:
        If split=True:
            tuple: (train_loader, val_loader) - DataLoaders for training and validation
        If split=False:
            torch.utils.data.DataLoader: Single DataLoader for the entire dataset
    """
    # Create the dataset
    dataset = POSTaggingDataset(tsv_file)
    
    # Function to create batches with padding
    def collate_fn(batch):
        # Sort batch by sentence length (descending) for pack_padded_sequence
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Separate words and tags
        words, tags = zip(*batch)
        
        # Get sequence lengths
        lengths = [len(x) for x in words]
        max_len = max(lengths)
        
        # Pad sequences
        padded_words = torch.zeros(len(words), max_len, dtype=torch.long)
        padded_tags = torch.zeros(len(tags), max_len, dtype=torch.long)
        
        for i, (word, tag) in enumerate(zip(words, tags)):
            padded_words[i, :len(word)] = word
            padded_tags[i, :len(tag)] = tag
        
        return padded_words, padded_tags, torch.LongTensor(lengths)
    
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


if __name__ == "__main__":
    # Example usage:
    # With split=True (returns train and validation loaders)
    train_loader, val_loader = build_dataloader(
        "train_raw.tsv", 
        "word_vocab.tsv", 
        "tag_vocab.tsv", 
        batch_size=32,
        split=True,
        val_ratio=0.1
    )


    # Check if any sample contains out-of-bounds indices
    for i, (words, _) in enumerate(train_loader.dataset.data):
        if max(words) >= len(train_loader.dataset):
            print(f"Sample {i} has out-of-bounds index: {max(words)}")

