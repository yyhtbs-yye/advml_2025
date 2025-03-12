from collections import Counter


# 2. Improved vocabulary building with frequency threshold
def build_vocab_with_threshold(sentences, min_freq=2, add_special_tokens=True):
    # Count word frequencies
    word_counter = Counter([word for sent in sentences for word in sent])
    
    # Start with special tokens
    if add_special_tokens:
        word_vocab = {
            "<PAD>": 0,      # For padding
            "<UNK>": 1,      # For unknown words
        }
    else:
        word_vocab = {}
    
    # Add words that appear at least min_freq times
    for word, count in word_counter.items():
        if count >= min_freq:
            word_vocab[word] = len(word_vocab)  # Removed + start_idx to avoid gaps
    
    return word_vocab, word_counter

# 3. Build character n-gram vocabulary for better OOV handling
def build_char_ngram_vocab(sentences, min_freq=5, ngram_range=(2, 4)):
    # Count character n-grams
    ngram_counter = Counter()
    for sent in sentences:
        for word in sent:
            # Add character n-grams
            for n in range(ngram_range[0], min(ngram_range[1] + 1, len(word) + 1)):
                for i in range(len(word) - n + 1):
                    ngram = "#" + word[i:i+n] + "#"  # Add boundary markers
                    ngram_counter[ngram] += 1
    
    # Build vocabulary with frequency threshold
    ngram_vocab = {"<PAD>": 0, "<UNK>": 1}
    for ngram, count in ngram_counter.items():
        if count >= min_freq:
            ngram_vocab[ngram] = len(ngram_vocab)
    
    return ngram_vocab, ngram_counter

# 4. Improved encoding function using both word and character n-grams
def encode_with_fallback(sentences, word_vocab, ngram_vocab, ngram_range=(2, 4)):
    encoded_data = []
    oov_count = 0
    total_words = 0
    
    for sent in sentences:
        word_indices = []
        subword_indices = []  # Store subword information for OOV words
        
        for word in sent:
            total_words += 1
            
            # Try direct word lookup first
            if word in word_vocab:
                word_indices.append(word_vocab[word])
            else:
                # Word is OOV, mark it and extract character n-grams
                oov_count += 1
                word_indices.append(word_vocab["<UNK>"])
                
                # Extract character n-grams as fallback representation
                word_ngrams = []
                for n in range(ngram_range[0], min(ngram_range[1] + 1, len(word) + 1)):
                    for i in range(len(word) - n + 1):
                        ngram = "#" + word[i:i+n] + "#"
                        if ngram in ngram_vocab:
                            word_ngrams.append(ngram_vocab[ngram])
                
                # Store the word's position and its n-grams
                if word_ngrams:
                    subword_indices.append((len(word_indices)-1, word_ngrams))
        
        # Store both word indices and subword information
        encoded_data.append((word_indices, subword_indices))
    
    oov_rate = oov_count / total_words if total_words > 0 else 0
    return encoded_data, oov_rate

