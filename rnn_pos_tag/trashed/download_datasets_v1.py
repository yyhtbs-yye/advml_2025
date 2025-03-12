import csv
import pandas as pd

import urllib.request, zipfile, io
import torch, torch.nn as nn, torch.optim as optim

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

# 2. Build vocabularies for words and tags
word_vocab = {"<UNK>": 1, "<PAD>": 0}  # unknown word token
for sent in train_sents:
    for w in sent:
        if w not in word_vocab:
            word_vocab[w] = len(word_vocab)

tag_vocab = {tag: i for i, tag in enumerate(sorted({t for tags in train_tags for t in tags}))}

# 3. Prepare data tensors (list of (word_indices, tag_indices) for each sentence)
def encode_dataset(sents, tags):
    data = []
    for words, pos_tags in zip(sents, tags):
        # convert to indices, unknown for OOV words
        word_idx = [word_vocab.get(w, 1) for w in words]
        tag_idx = [tag_vocab[t] for t in pos_tags]
        data.append((word_idx, tag_idx))
    return data

train_data = encode_dataset(train_sents, train_tags)
dev_data   = encode_dataset(dev_sents, dev_tags)
test_data  = encode_dataset(test_sents, test_tags)

# This function saves each sentence and its tags as a full row with tab delimiter
def save_raw_data_to_csv(sentences, tags, filename):
    # Create a list to store data
    data = []
    
    # Iterate through sentences and their tags
    for sent, sent_tags in zip(sentences, tags):
        # Ensure both lists have the same length
        assert len(sent) == len(sent_tags), "Mismatch between words and tags length"
        
        # Join words and tags into space-separated strings
        sentence = " ".join(sent)
        pos_tags = " ".join(sent_tags)
        
        # Add the full sentence and its tags as a row
        data.append({"sentence": sentence, "pos_tags": pos_tags})
    
    # Convert to DataFrame and save with tab delimiter
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, sep='\t')
    print(f"Saved {len(data)} sentences to {filename}")

# This function saves the encoded data (with word and tag indices)
def save_encoded_data_to_csv(data, word_vocab, tag_vocab, filename):
    # Create reverse mappings for readability
    id_to_word = {idx: word for word, idx in word_vocab.items()}
    id_to_tag = {idx: tag for tag, idx in tag_vocab.items()}
    
    # Create a list to store data
    rows = []
    
    # Iterate through the encoded data
    for sent_idx, (word_indices, tag_indices) in enumerate(data):
        # Convert indices to words and tags
        words = [id_to_word.get(idx, "<UNK>") for idx in word_indices]
        tags = [id_to_tag.get(idx, "UNK") for idx in tag_indices]
        
        # Join words and tags into space-separated strings
        sentence = " ".join(words)
        pos_tags = " ".join(tags)
        
        # Convert indices to space-separated strings for reference
        word_indices_str = " ".join(map(str, word_indices))
        tag_indices_str = " ".join(map(str, tag_indices))
        
        rows.append({
            "sentence": sentence,
            "pos_tags": pos_tags,
            "word_indices": word_indices_str,
            "tag_indices": tag_indices_str
        })
    
    # Convert to DataFrame and save with tab delimiter
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, sep='\t')
    print(f"Saved {len(rows)} sentences to {filename}")

# Save vocabularies as CSV files
def save_vocabs_to_csv(word_vocab, tag_vocab):
    # Save word vocabulary with tab delimiter
    pd.DataFrame({"word": list(word_vocab.keys()), "index": list(word_vocab.values())}).to_csv(
        "datasets/word_vocab.tsv", index=False, sep='\t'
    )
    
    # Save tag vocabulary with tab delimiter
    pd.DataFrame({"tag": list(tag_vocab.keys()), "index": list(tag_vocab.values())}).to_csv(
        "datasets/tag_vocab.tsv", index=False, sep='\t'
    )
    
    print(f"Saved vocabularies: {len(word_vocab)} words, {len(tag_vocab)} tags")

# Save the raw datasets
save_raw_data_to_csv(train_sents, train_tags, "datasets/train_raw.tsv")
save_raw_data_to_csv(dev_sents, dev_tags, "datasets/dev_raw.tsv")
save_raw_data_to_csv(test_sents, test_tags, "datasets/test_raw.tsv")

# Save the encoded datasets
save_encoded_data_to_csv(train_data, word_vocab, tag_vocab, "datasets/train_encoded.tsv")
save_encoded_data_to_csv(dev_data, word_vocab, tag_vocab, "datasets/dev_encoded.tsv")
save_encoded_data_to_csv(test_data, word_vocab, tag_vocab, "datasets/test_encoded.tsv")

# Save the vocabularies
save_vocabs_to_csv(word_vocab, tag_vocab)