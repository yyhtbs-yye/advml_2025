import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

def nlp_transform(sentence, normalization=None, remove_stopwords=False):
    """
    Transform a sentence by applying various NLP techniques.
    
    Parameters:
    -----------
    sentence : str
        The input sentence to transform
    normalization : str or None
        The type of normalization to apply: 'lower', 'lemma', or None
    remove_stopwords : bool
        Whether to remove stopwords from the sentence
        
    Returns:
    --------
    list
        A list of processed tokens from the sentence
    """
    if not sentence or not isinstance(sentence, str):
        return []
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    sentence = sentence.translate(translator)
    

    # Clean the text using regex
    # 1. Remove URLs
    sentence = re.sub(r'https?://\S+|www\.\S+', '', sentence)
    
    # 2. Remove email addresses
    sentence = re.sub(r'\S+@\S+', '', sentence)
    
    # 3. Remove emojis - basic pattern for common emoji unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE
    )
    sentence = emoji_pattern.sub(r'', sentence)
    
    # 4. Remove HTML tags
    sentence = re.sub(r'<.*?>', '', sentence)
    
    # 5. Remove punctuation and special characters
    sentence = re.sub(r'[^\w\s]', '', sentence)
    
    # 6. Remove extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    
    # 7. Remove numbers
    sentence = re.sub(r'\d+', '', sentence)

    # Tokenize the sentence
    tokens = word_tokenize(sentence.lower())
    
    # Apply normalization if specified
    if normalization == 'lemma':
        lemmatizer = WordNetLemmatizer()
        # First lowercase for better lemmatization
        tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    
    # Remove stopwords if specified
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.lower() not in stop_words]
    
    return tokens

# Example usage with partial function application
from functools import partial

# This is how you might use the function with config values
def create_transform_from_config(config):
    return partial(
        nlp_transform,
        normalization=config["normalization"],
        remove_stopwords=config["remove_stopwords"]
    )

# Example use in dataset
# nlp_transform = create_transform_from_config(config)
# dataset = SentimentAnalysisDataset(root_folder, vocab_file, nlp_transform)