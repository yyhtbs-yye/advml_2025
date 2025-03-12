from helpers.acl_imdb_dataset import build_dataloader
import json

# Example usage
if __name__ == "__main__":
    # Test the entire pipeline
    dataset_path = "/home/admyyh/python_workspace/advml/rnn_sa/aclImdb/train"
    vocab_path = "/home/admyyh/python_workspace/advml/rnn_sa/word_vocab_norm_None_stop_False.json"
    
    print("\nStep 1: Creating dataloader...")
    train_dataloader, validation_dataloader = build_dataloader(
        root_folder=dataset_path, vocab_file=vocab_path,
        batch_size=8,  # Using batch size 8 for testing
        shuffle=False,
        split=True, val_ratio=0.1, random_seed=42,
    )
    
    # Test the dataloader
    print(f"\nDataloader created with {len(train_dataloader.dataset)} samples")
    
    # Get a batch
    print("\nTesting batch extraction:")
    for batch_idx, (features, labels, lengths) in enumerate(train_dataloader):
        print(f"Batch {batch_idx+1}:")
        print(f"  - Features shape: {features.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Sequence lengths: {lengths}")

        # Load vocabulary from json file
        with open("word_vocab_norm_None_stop_False.json", 'r') as f:
            vocab_data = json.load(f)
        

        # Print sample
        sample_idx = 0
        print(f"\nSample from batch (item {sample_idx}):")
        print(f"  - Label: {labels[sample_idx].item()} ({'Positive' if labels[sample_idx].item() == 1 else 'Negative'})")
        print(f"  - Sequence length: {lengths[sample_idx].item()}")
        print(f"  - First 10 tokens (as indices): {features[sample_idx][:10].tolist()}")
        
        # Convert indices back to words for demonstration
        idx_to_word = {data["index"]: word for word, data in vocab_data["word_vocab"].items()}
        words = [idx_to_word.get(idx.item(), "<UNK>") for idx in features[sample_idx][:10]]
        print(f"  - First 10 tokens (as words): {words}")
        
        # Only process the first batch for demonstration
        if batch_idx == 0:
            break
    
    print("\nTest complete!")
