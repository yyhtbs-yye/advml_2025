import torch
import pandas as pd
import os
import json
from functools import partial
from torch.nn.utils.rnn import pad_sequence
import random

class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data_chunks, vocab_file, max_length=2048):
        """
        Character-level dataset for language modeling
        
        Args:
            data_chunks (list): List of text chunks
            vocab_file (str): Path to vocabulary JSON file
            max_length (int): Maximum sequence length
        """
        self.data_chunks = data_chunks
        self.max_length = max_length
        
        # Load vocabulary from json file
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
        
        # Create character to index mapping
        self.char_to_idx = {char: data["index"] for char, data in vocab_data["letter_vocab"].items()}
        self.idx_to_char = {data["index"]: char for char, data in vocab_data["letter_vocab"].items()}
        
        # PAD and UNK tokens
        self.pad_idx = self.char_to_idx.get("<PAD>", 0)
        self.unk_idx = self.char_to_idx.get("<UNK>", 1)
    
    def __len__(self):
        return len(self.data_chunks)
    
    def __getitem__(self, idx):
        # Get text chunk
        text_chunk = self.data_chunks[idx]
        
        # Convert to character indices
        char_indices = []
        for char in text_chunk:
            # Use UNK index for characters not in vocabulary
            char_indices.append(self.char_to_idx.get(char, self.unk_idx))
        
        # Limit length if needed
        if len(char_indices) > self.max_length:
            start_idx = random.randint(0, len(char_indices) - self.max_length)
            char_indices = char_indices[start_idx:start_idx + self.max_length]
        
        # Convert to tensor
        input_tensor = torch.tensor(char_indices[:-1], dtype=torch.long)
        target_tensor = torch.tensor(char_indices[1:], dtype=torch.long)
        
        return input_tensor, target_tensor

def chunk_text_file(file_path, chunk_size=200, overlap=50):
    """
    Read a text file and split it into overlapping chunks
    
    Args:
        file_path (str): Path to the text file
        chunk_size (int): Size of each chunk
        overlap (int): Overlap between consecutive chunks
        
    Returns:
        list: List of text chunks
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
            
        return chunks
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

def prepare_dataset(root_folder, output_folder="dataset", chunk_size=200, overlap=50, max_files=50):
    """
    Prepare a dataset by chunking text files
    
    Args:
        root_folder (str): Folder containing text files
        output_folder (str): Folder to save chunked data
        chunk_size (int): Size of each chunk
        overlap (int): Overlap between consecutive chunks
        max_files (int): Maximum number of files to process
        
    Returns:
        list: List of all text chunks
    """
    import os
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    all_chunks = []
    
    # Get all text files
    text_files = []
    for file in os.listdir(root_folder):
        file_path = os.path.join(root_folder, file)
        if file.endswith('.txt') and os.path.isfile(file_path):
            text_files.append(file_path)
    
    # Limit the number of files if needed
    if max_files > 0 and len(text_files) > max_files:
        text_files = text_files[:max_files]
    
    # Process each file
    for i, file_path in enumerate(text_files):
        file_chunks = chunk_text_file(file_path, chunk_size, overlap)
        all_chunks.extend(file_chunks)
        
        # Save chunks to output folder
        filename = os.path.basename(file_path)
        with open(os.path.join(output_folder, f"chunked_{i}_{filename}"), 'w', encoding='utf-8') as f:
            f.write('\n---CHUNK---\n'.join(file_chunks))
        
        print(f"Processed {i+1}/{len(text_files)}: {filename} - {len(file_chunks)} chunks")
    
    # Save all chunks to a single file
    with open(os.path.join(output_folder, "all_chunks.txt"), 'w', encoding='utf-8') as f:
        f.write('\n---CHUNK---\n'.join(all_chunks))
    
    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks

def build_dataloader(data_chunks, vocab_file, batch_size, max_length=100, shuffle=True,
                     split=False, val_ratio=0.1, random_seed=42):
    """
    Build PyTorch DataLoader for character-level language modeling
    
    Args:
        data_chunks (list): List of text chunks
        vocab_file (str): Path to vocabulary JSON file
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        shuffle (bool): Whether to shuffle data
        split (bool): Whether to split into training and validation sets
        val_ratio (float): Ratio of validation set
        random_seed (int): Random seed for reproducibility
        
    Returns:
        DataLoader or tuple of DataLoaders: PyTorch DataLoader(s)
    """
    # Create dataset
    dataset = CharDataset(data_chunks, vocab_file, max_length)
    
    def collate_fn(batch):
        # Separate inputs and targets
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        
        # Pad sequences to the maximum length in this batch
        inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=dataset.pad_idx)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=dataset.pad_idx)
        
        # Store original sequence lengths
        lengths = torch.tensor([len(inp) for inp in inputs], dtype=torch.long)
        
        return inputs_padded, targets_padded, lengths
    
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

# Example usage
if __name__ == "__main__":
    # Prepare dataset by chunking text files
    data_chunks = prepare_dataset(
        root_folder="./",
        output_folder="chunked_data",
        chunk_size=200,
        overlap=50,
        max_files=10  # Limit to 10 files for testing
    )
    
    # Build dataloader with training/validation split
    train_loader, val_loader = build_dataloader(
        data_chunks=data_chunks,
        vocab_file="letter_vocab_min_freq.json",
        batch_size=32,
        max_length=100,
        shuffle=True,
        split=True,
        val_ratio=0.1
    )
    
    # Example batch
    for inputs, targets, lengths in train_loader:
        print(f"Batch size: {inputs.shape[0]}")
        print(f"Sequence length: {inputs.shape[1]}")
        print(f"Input tensor shape: {inputs.shape}")
        print(f"Target tensor shape: {targets.shape}")
        print(f"Length tensor shape: {lengths.shape}")
        break