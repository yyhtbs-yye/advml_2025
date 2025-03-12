from helpers.nlp_normalization import nlp_transform

def build_vocabulary(root_folder, min_freq=1, special_tokens=None, normalization="lemma", remove_stopwords=False):
    """
    Build a vocabulary from text files in the given folder structure.
    
    Args:
        root_folder (str): Path to the root folder containing text subfolders
        min_freq (int): Minimum frequency for a word to be included in the vocabulary
        special_tokens (list): List of special tokens to include in the vocabulary
        normalization (str): Normalization technique - "stem", "lemma", or "none"
        remove_stopwords (bool): Whether to remove stopwords
        
    Returns:
        dict: Vocabulary dictionary ready to be saved as JSON
    """
    import os
    from collections import Counter
    
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>"]
    
    # Get all text files
    all_files = []
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.txt'):
                    all_files.append(os.path.join(subdir_path, file))
    
    # Process all files to count words
    word_counter = Counter()
    original_to_normalized = {}  # To keep track of original forms
    
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
                
                # Apply NLP transformations
                tokens = nlp_transform(
                    text, 
                    normalization=normalization,
                    remove_stopwords=remove_stopwords
                )
                
                # Keep track of original forms (for reference)
                for token in tokens:
                    if token not in original_to_normalized:
                        original_to_normalized[token] = set()
                    # Only add if it's different from the normalized form
                    if token.lower() != token:
                        original_to_normalized[token].add(token.lower())
                
                word_counter.update(tokens)
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Filter words by frequency
    filtered_words = [word for word, count in word_counter.items() 
                     if count >= min_freq]
    
    # Create the vocabulary dictionary
    word_vocab = {}
    
    # Add special tokens first
    for i, token in enumerate(special_tokens):
        word_vocab[token] = {
            "index": i,
            "frequency": 0
        }
    
    # Add the rest of the words
    for i, word in enumerate(filtered_words):
        # Skip if the word is already in special tokens
        if word in word_vocab:
            continue
        
        word_vocab[word] = {
            "index": i + len(special_tokens),
            "frequency": word_counter[word],
            "original_forms": list(original_to_normalized.get(word, set()))
        }
    
    return {"word_vocab": word_vocab}

def save_vocabulary(vocab, output_file):
    """Save the vocabulary to a JSON file"""
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)

# Example usage
if __name__ == "__main__":
    # Example usage
    sample_text = "I was a human being, and I wasn't a alien. I am a alien, and I am no a human being."
    tokens = nlp_transform(sample_text, normalization=None, remove_stopwords=False)
    print(tokens)  # Output: ['run', 'quickly', 'park', 'noticed', 'beautiful', 'flower']
    config = {
        "min_freq": 2,
        "normalization": None,
        "remove_stopwords": False,
    }
    filename = f"word_vocab_norm_{config['normalization']}_stop_{config['remove_stopwords']}.json"
    # Build vocabulary example
    vocab = build_vocabulary(
        "aclImdb/train",
        min_freq=config["min_freq"],
        normalization=config["normalization"],
        remove_stopwords=config["remove_stopwords"]
    )
    vocab["config"] = config
    save_vocabulary(vocab, filename)