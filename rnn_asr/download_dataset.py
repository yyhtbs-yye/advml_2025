import urllib.request, os, zipfile, json
import torchaudio
import tarfile
import torch
import shutil
import librosa

# 1. Download and extract the AN4 dataset (audio + transcripts)
url = "https://huggingface.co/datasets/espnet/an4/resolve/main/an4_sphere.tar.gz"
fn = "an4.tar.gz"
if not os.path.exists(fn):
    urllib.request.urlretrieve(url, fn)
    with tarfile.open(fn, 'r:gz') as tar:
        tar.extractall()  # extracts to ./an4/ directory

# 2. Read transcripts and file lists
train_paths, train_texts = [], []
test_paths, test_texts = [], []

def load_an4_manifest(trans_file, fileid_file, wav_dir):
    paths, texts = [], []
    # Read all lines of transcription and file IDs
    trans_lines = open(trans_file).read().strip().splitlines()
    ids = [line.strip() for line in open(fileid_file)]
    for line, fid in zip(trans_lines, ids):
        # Transcription format: "TRANSCRIPT TEXT (<fileid>)"
        text = line.split("(", 1)[0].strip()  # get text before '('
        # Construct wav path: wav_dir + fileid + ".wav"
        wav_path = os.path.join(wav_dir, fid + ".sph")
        paths.append(wav_path)
        # normalize text: uppercase and remove punctuation (AN4 is already uppercase)
        text = text.strip().upper()
        texts.append(text)
    return paths, texts

train_paths, train_texts = load_an4_manifest("an4/etc/an4_train.transcription",
                                            "an4/etc/an4_train.fileids",
                                            "an4/wav/")
test_paths, test_texts = load_an4_manifest("an4/etc/an4_test.transcription",
                                          "an4/etc/an4_test.fileids",
                                          "an4/wav/")

# 3. Build character vocabulary (including blank for CTC and space as character)
chars = set("".join(train_texts))
chars.update([" "])  # ensure space included if not present
chars = sorted(list(chars))
blank_char = "_"  # CTC blank
chars.insert(0, blank_char)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# 3a. Build unified word vocabulary
all_words = []
for text in train_texts + test_texts:
    words = text.split()
    all_words.extend(words)

unique_words = sorted(set(all_words))
# Add special tokens
special_tokens = ["<UNK>", "<PAD>", "<BOS>", "<EOS>"]
vocabulary = special_tokens + unique_words

# Create word-to-id mapping
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

train_waveforms = []
for wav_path in train_paths:
    sr = 16000
    waveform, sr = librosa.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(torch.tensor(waveform), sr, 16000)

    train_waveforms.append(waveform) 

test_waveforms = []
for wav_path in test_paths:
    waveform, sr = librosa.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(torch.tensor(waveform), sr, 16000)
    test_waveforms.append(waveform)

# 6. Create output directories
output_base_dir = "datasets"
train_dir = os.path.join(output_base_dir, "train")
test_dir = os.path.join(output_base_dir, "test")

# Create directories if they don't exist
for directory in [output_base_dir, train_dir, test_dir]:
    if os.path.exists(directory) and directory != output_base_dir:
        shutil.rmtree(directory)  # Clean up if directory exists
    os.makedirs(directory, exist_ok=True)

# 7. Save vocabularies
# Character vocabulary for CTC
char_vocab = {
    "char_to_idx": char_to_idx,
    "idx_to_char": {str(k): v for k, v in idx_to_char.items()},  # Convert int keys to strings for JSON
    "blank_char": blank_char
}
with open(os.path.join(output_base_dir, "char_vocab.json"), "w") as f:
    json.dump(char_vocab, f, indent=2)

# Word vocabulary
word_vocab = {
    "word_to_idx": word_to_idx,
    "idx_to_word": {str(k): v for k, v in idx_to_word.items()},  # Convert int keys to strings for JSON
    "special_tokens": special_tokens
}
with open(os.path.join(output_base_dir, "unified_vocab.json"), "w") as f:
    json.dump(word_vocab, f, indent=2)

# 8. Function to save waveforms and texts
def save_dataset(waveforms, texts, output_dir):
    for i, (waveform, text) in enumerate(zip(waveforms, texts)):
        # Convert tensor to numpy and then to list for JSON serialization
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.numpy()
        else:
            waveform_np = waveform
        
        # Convert text to word IDs and character IDs
        words = text.split()
        word_ids = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in words]
        char_ids = [char_to_idx[c] for c in text]
        
        # Create a dictionary with waveforms and metadata
        data = {
            "waveform": waveform_np.tolist(),
            "text": text,
            "char_ids": char_ids,  # Character-level encoding for CTC
            "word_ids": word_ids   # Word-level encoding for language modeling
        }
        
        # Save as JSON
        json_path = os.path.join(output_dir, f"sample_{i:04d}.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        # Print progress
        if i % 100 == 0:
            print(f"Saved {i+1} samples...")

# 9. Save all datasets
print(f"Saving training set ({len(train_waveforms)} samples)...")
save_dataset(train_waveforms, train_texts, train_dir)

print(f"Saving test set ({len(test_waveforms)} samples)...")
save_dataset(test_waveforms, test_texts, test_dir)

# 10. Save dataset statistics
stats = {
    "dataset_name": "AN4",
    "num_train_samples": len(train_waveforms),
    "num_test_samples": len(test_waveforms),
    "char_vocabulary_size": len(chars),
    "word_vocabulary_size": len(vocabulary),
    "chars": chars,
    "words": vocabulary[:20] + ["..."] + vocabulary[-20:] if len(vocabulary) > 40 else vocabulary
}

with open(os.path.join(output_base_dir, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print(f"Preprocessing complete. Data saved to {output_base_dir}/")
print(f"- Training samples: {len(train_waveforms)}")
print(f"- Test samples: {len(test_waveforms)}")
print(f"- Character vocabulary size: {len(chars)}")
print(f"- Word vocabulary size: {len(vocabulary)}")
print(f"- Word vocabulary file: {os.path.join(output_base_dir, 'unified_vocab.json')}")