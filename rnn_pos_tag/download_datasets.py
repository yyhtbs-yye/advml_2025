import csv
import pandas as pd
import urllib.request, zipfile, io
import torch, torch.nn as nn, torch.optim as optim
import random
import os
import json

from utils.nlp_preproc import *

# Create datasets directory if it doesn't exist
os.makedirs("datasets", exist_ok=True)

# 1. Download and read the UD English EWT dataset (train/dev/test in CoNLL-U format)
url = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-{split}.conllu"
datasets = {}
for split in ["train", "dev", "test"]:
    data = urllib.request.urlopen(url.format(split=split)).read().decode('utf-8').strip().splitlines()
    sentences, tags = [], []
    sent_words, sent_tags = [], []
    for line in data:
        if not line or line.startswith("#"):
            # end of sentence
            if sent_words:
                sentences.append(sent_words); tags.append(sent_tags)
                sent_words, sent_tags = [], []
        else:
            cols = line.split('\t')
            if cols[0].isdigit():  # normal token line
                word = cols[1]; pos = cols[3]  # form and UPOS
                sent_words.append(word); sent_tags.append(pos)
    datasets[split] = (sentences, tags)

train_sents, train_tags = datasets["train"]
dev_sents, dev_tags = datasets["dev"]
test_sents, test_tags = datasets["test"]

# Create main word vocabulary with frequency threshold
word_vocab, word_counter = build_vocab_with_threshold(train_sents, min_freq=5)


# Create character n-gram vocabulary
ngram_vocab, ngram_counter = build_char_ngram_vocab(train_sents, min_freq=10)

# 5. Apply encoding to each dataset
train_encoded, train_oov_rate = encode_with_fallback(train_sents, word_vocab, ngram_vocab)
dev_encoded, dev_oov_rate = encode_with_fallback(dev_sents, word_vocab, ngram_vocab)
test_encoded, test_oov_rate = encode_with_fallback(test_sents, word_vocab, ngram_vocab)

print(f"OOV rates: Train: {train_oov_rate:.2%}, Dev: {dev_oov_rate:.2%}, Test: {test_oov_rate:.2%}")

# 6. Tag vocabulary building (unchanged)
tag_vocab = {tag: i for i, tag in enumerate(sorted({t for tags in train_tags for t in tags}))}

# 7. Create tag indices for each sentence
def encode_tags(tag_sequences, tag_vocab):
    return [[tag_vocab[t] for t in tags] for tags in tag_sequences]

train_tag_indices = encode_tags(train_tags, tag_vocab)
dev_tag_indices = encode_tags(dev_tags, tag_vocab)
test_tag_indices = encode_tags(test_tags, tag_vocab)

# 8. Save vocabulary statistics to a report file
with open("datasets/vocab_statistics.txt", "w") as f:
    f.write(f"Dataset Statistics:\n")
    f.write(f"  Train: {len(train_sents)} sentences, {sum(len(s) for s in train_sents)} tokens\n")
    f.write(f"  Dev: {len(dev_sents)} sentences, {sum(len(s) for s in dev_sents)} tokens\n")
    f.write(f"  Test: {len(test_sents)} sentences, {sum(len(s) for s in test_sents)} tokens\n\n")
    
    f.write(f"Vocabulary Statistics:\n")
    f.write(f"  Word vocabulary size: {len(word_vocab)}\n")
    f.write(f"  Character n-gram vocabulary size: {len(ngram_vocab)}\n")
    f.write(f"  POS tag vocabulary size: {len(tag_vocab)}\n\n")
    
    f.write(f"OOV Rates:\n")
    f.write(f"  Train: {train_oov_rate:.2%}\n")
    f.write(f"  Dev: {dev_oov_rate:.2%}\n")
    f.write(f"  Test: {test_oov_rate:.2%}\n\n")
    
    f.write(f"Most common words:\n")
    for word, count in word_counter.most_common(20):
        f.write(f"  {word}: {count}\n")
    
    f.write(f"\nMost common character n-grams:\n")
    for ngram, count in ngram_counter.most_common(20):
        f.write(f"  {ngram}: {count}\n")

# 9. Create a unified vocabulary and save to a single JSON file
def create_and_save_unified_vocab():
    unified_vocab = {
        "word_vocab": {word: {"index": idx, "frequency": word_counter.get(word, 0)} 
                      for word, idx in word_vocab.items()},
        "ngram_vocab": {ngram: {"index": idx, "frequency": ngram_counter.get(ngram, 0)}
                       for ngram, idx in ngram_vocab.items()},
        "tag_vocab": {tag: {"index": idx} for tag, idx in tag_vocab.items()},
        "metadata": {
            "word_vocab_size": len(word_vocab),
            "ngram_vocab_size": len(ngram_vocab),
            "tag_vocab_size": len(tag_vocab),
            "oov_rates": {
                "train": train_oov_rate,
                "dev": dev_oov_rate,
                "test": test_oov_rate
            }
        }
    }
    
    # Save as JSON
    with open("datasets/unified_vocab.json", "w", encoding="utf-8") as f:
        json.dump(unified_vocab, f, indent=2, ensure_ascii=False)
    
    print("Saved unified vocabulary to datasets/unified_vocab.json")
    
    # Also save as TSV for compatibility
    with open("datasets/unified_vocab.tsv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["vocab_type", "token", "index", "frequency"])
        
        # Write word vocab
        for word, idx in sorted(word_vocab.items(), key=lambda x: x[1]):
            writer.writerow(["word", word, idx, word_counter.get(word, 0)])
        
        # Write ngram vocab
        for ngram, idx in sorted(ngram_vocab.items(), key=lambda x: x[1]):
            writer.writerow(["ngram", ngram, idx, ngram_counter.get(ngram, 0)])
        
        # Write tag vocab
        for tag, idx in sorted(tag_vocab.items(), key=lambda x: x[1]):
            writer.writerow(["tag", tag, idx, ""])
    
    # Save separate tag vocabulary file
    with open("datasets/tag_vocab.json", "w", encoding="utf-8") as f:
        tag_data = {
            "tag_vocab": {tag: idx for tag, idx in tag_vocab.items()},
            "inverse_tag_vocab": {str(idx): tag for tag, idx in tag_vocab.items()},
            "metadata": {
                "tag_vocab_size": len(tag_vocab),
                "tags": list(tag_vocab.keys())
            }
        }
        json.dump(tag_data, f, indent=2, ensure_ascii=False)
    
    # Save tag vocabulary as TSV
    with open("datasets/tag_vocab.tsv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["tag", "index"])
        for tag, idx in sorted(tag_vocab.items(), key=lambda x: x[1]):
            writer.writerow([tag, idx])
    
    print("Saved tag vocabulary to datasets/tag_vocab.json and datasets/tag_vocab.tsv")

create_and_save_unified_vocab()

# 10. Save encoded dataset with both word and subword information
def save_encoded_dataset(split_name, sentences, tags, encoded_data, tag_indices):
    filename = f"datasets/{split_name}_improved.tsv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "sentence", "pos_tags", 
            "word_indices", "tag_indices", 
            "has_oov", "oov_positions", "subword_info"
        ])
        
        for i, ((word_indices, subword_indices), tag_idx) in enumerate(zip(encoded_data, tag_indices)):
            # Original text
            original_sent = " ".join(sentences[i])
            original_tags = " ".join(tags[i])
            
            # Encoded indices
            word_indices_str = " ".join(map(str, word_indices))
            tag_indices_str = " ".join(map(str, tag_idx))
            
            # OOV information
            has_oov = 1 if subword_indices else 0
            oov_positions = " ".join(str(pos) for pos, _ in subword_indices)
            
            # Subword information (format: "pos1:ngram1,ngram2;pos2:ngram3,ngram4")
            subword_info = ";".join(
                f"{pos}:{','.join(map(str, ngrams))}" 
                for pos, ngrams in subword_indices
            )
            
            writer.writerow([
                original_sent, original_tags,
                word_indices_str, tag_indices_str,
                has_oov, oov_positions, subword_info
            ])
    
    print(f"Saved {len(sentences)} sentences to {filename}")

# Save the improved encoded datasets
save_encoded_dataset("train", train_sents, train_tags, train_encoded, train_tag_indices)
save_encoded_dataset("dev", dev_sents, dev_tags, dev_encoded, dev_tag_indices)
save_encoded_dataset("test", test_sents, test_tags, test_encoded, test_tag_indices)

# 11. Function to load unified vocabulary from JSON (for reference in future code)
def load_unified_vocab(filepath="datasets/unified_vocab.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        unified_vocab = json.load(f)
    
    # Convert the nested dictionaries back to simple index mappings for direct use
    word_vocab = {word: data["index"] for word, data in unified_vocab["word_vocab"].items()}
    ngram_vocab = {ngram: data["index"] for ngram, data in unified_vocab["ngram_vocab"].items()}
    tag_vocab = {tag: data["index"] for tag, data in unified_vocab["tag_vocab"].items()}
    
    return word_vocab, ngram_vocab, tag_vocab, unified_vocab["metadata"]

print("All datasets and unified vocabulary have been saved!")