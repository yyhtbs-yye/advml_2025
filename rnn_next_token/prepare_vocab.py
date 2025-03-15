
def build_vocabulary(root_folder, min_freq=1, special_tokens=None):
    """
    Build a letter-level vocabulary from text files in the given folder structure.
    
    Args:
        root_folder (str): Path to the root folder containing text subfolders
        min_freq (int): Minimum frequency for a letter to be included in the vocabulary
        special_tokens (list): List of special tokens to include in the vocabulary        
    Returns:
        dict: Vocabulary dictionary ready to be saved as JSON
    """
    import os
    from collections import Counter
    
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>"]
    
    # Get all text files
    all_files = []
    for file in os.listdir(root_folder):
        file_path = os.path.join(root_folder, file)
        if file.endswith('.txt') and os.path.isfile(file_path):
            all_files.append(file_path)
    
    # Process all files to count letters
    letter_counter = Counter()
    
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
                
                # Extract individual letters instead of words
                letters = list(text)
                letter_counter.update(letters)
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Filter letters by frequency
    filtered_letters = [letter for letter, count in letter_counter.items() 
                       if count >= min_freq]
    
    # Create the vocabulary dictionary
    letter_vocab = {}
    
    # Add special tokens first
    for i, token in enumerate(special_tokens):
        letter_vocab[token] = {
            "index": i,
            "frequency": 0
        }
    
    # Add the rest of the letters
    for i, letter in enumerate(filtered_letters):
        # Skip if the letter is already in special tokens
        if letter in letter_vocab:
            continue
        
        letter_vocab[letter] = {
            "index": i + len(special_tokens),
            "frequency": letter_counter[letter]
        }
    
    return {"letter_vocab": letter_vocab}

def save_vocabulary(vocab, output_file):
    """Save the vocabulary to a JSON file"""
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)

def letter_transform(text):
    """
    Transform text into letter-level tokens.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of individual letters
    """
    return list(text)

# Example usage
if __name__ == "__main__":
    # Example usage
    sample_text = "I was a human being, and I wasn't a alien. I am a alien, and I am no a human being."
    letters = letter_transform(sample_text)
    print(letters)  # Output: ['I', ' ', 'w', 'a', 's', ' ', 'a', ...]
    
    
    filename = f"letter_vocab_min_freq.json"
    
    # Build vocabulary example
    vocab = build_vocabulary(
        "./",
        min_freq=1,
    )
    
    save_vocabulary(vocab, filename)