from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # TODO: Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    examples: List[SentimentExample] = []
    
    with open(infile, "r", encoding='utf-8') as f:
        for line in f:
            #Check there are only two values to unpack: sentence and label
            if len(line.strip().split("\t")) == 2:
                sentence, label = line.strip().split("\t") # Split the line into sentence and label
                words = tokenize(sentence) # Tokenize the sentence
                label = int(label)
                examples.append(SentimentExample(words, label))

    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}
    word_counts = Counter()

    for example in examples:
        word_counts.update(example.words)
    
    #word_counts = Counter({'manzana': 3, 'pera': 2, 'naranja': 1})
    vocab = {word:idx for idx, word in enumerate(word_counts.keys())}
    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full
    bow: torch.Tensor = torch.zeros(len(vocab), dtype=torch.float32) #Xd ∈ R^|V|, where |V| is the size of the vocabulary
    word_counts = Counter(text) #Count the frequency of each word in the text
    if binary:
        for word, idx in vocab.items():
            if word in word_counts:
                bow[idx] = 1 #presence or absence of the word in the text (binary represeentation)
    else:
        for word, idx in vocab.items():
            bow[idx] = word_counts[word]

    return bow

