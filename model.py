"""
NumPy Text Classifier from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - clean_text
def clean_text(text: str) -> str:
    text = text.lower()

    cleaned = ""
    for ch in text:
        if ch.isalpha():
            cleaned += ch
        else:
            cleaned += " "

    return cleaned.strip()

# Step 2 - tokenize
def tokenize(text: str) -> list:
    return text.split()

# Step 3 - tokenize_corpus
def tokenize_corpus(texts: list) -> list:
    # TODO: Apply clean_text and tokenize to every document so the full corpus becomes a list of token lists.
    corpus_tokens=[]

    for text in texts:
        cleaned_text=clean_text(text)
        tokens= tokenize(cleaned_text)
        corpus_tokens.append(tokens)
    return corpus_tokens

# Step 4 - split_train_val_test_indices
# def split_train_val_test_indices(n_samples: int, val_fraction: float, test_fraction: float, seed: int = 0) -> tuple:
#     # TODO: Produce shuffled index arrays that partition n_samples into train/val/test
    
#     np.random.seed(seed)
#     indices= np.arange(n_samples)
#     np.random.shuffle(indices)
#     n_val=  int(n_samples* val_fraction)
#     n_test= int(n_samples*test_fraction)
#     n_train = n_samples - n_val - n_test
#     train_idx=indices[:n_train]
#     val_idx = indices[n_train:n_train + n_val]
#     test_idx = indices[n_train + n_val:]
#     return train_idx, val_idx, test_idx



def split_train_val_test_indices(
    n_samples: int,
    val_fraction: float,
    test_fraction: float,
    seed: int = 0
) -> tuple:
    # Set NumPy's random seed so that we get the same shuffle every time
    np.random.seed(seed)

    # Create an array containing indices from 0 to n_samples - 1
    # Example: n_samples = 10 -> [0, 1, 2, ..., 9]
    indices = np.arange(n_samples)

    # Shuffle the indices randomly
    # The original indices array is shuffled in-place
    np.random.shuffle(indices)

    # Calculate the number of validation samples
    # int() truncates the decimal part
    # Example: 10 * 0.2 = 2.0 -> 2
    n_val = int(n_samples * val_fraction)

    # Calculate the number of test samples
    n_test = int(n_samples * test_fraction)

    # The remaining samples go to the training set
    n_train = n_samples - n_val - n_test

    # Take the first n_train shuffled indices for training
    train_idx = indices[:n_train]

    # Take the next n_val indices for validation
    val_idx = indices[n_train:n_train + n_val]

    # Take the remaining indices for testing
    test_idx = indices[n_train + n_val:]

    # Return train, validation, and test indices in that order
    return train_idx, val_idx, test_idx

# Step 5 - count_word_frequencies
def count_word_frequencies(tokenized_docs: list) -> dict:
    # TODO: Return a dict mapping each unique token to its total count...
    word_counts={}
    for doc in tokenized_docs:
        
        for token in doc:
            if token in word_counts:
                word_counts[token]+=1
            else:
                word_counts[token]=1
    return word_counts

# Step 6 - build_vocabulary
def build_vocabulary(word_counts: dict, max_size: int) -> dict:
    # Sort all words:
    # 1. Higher frequency comes first -> -item[1]
    # 2. If frequency is same, alphabetical order -> item[0]
    sorted_words = sorted(
        word_counts.items(),
        key=lambda item: (-item[1], item[0])
    )

    # Keep only the top max_size words
    top_words = sorted_words[:max_size]

    # Create vocabulary dictionary
    # Most frequent word gets index 0,
    # second most frequent gets index 1, and so on
    vocabulary = {}

    for index, (word, count) in enumerate(top_words):
        vocabulary[word] = index

    # Return word -> integer index mapping
    return vocabulary

# Step 7 - tokens_to_bow
def tokens_to_bow(tokens: list, vocab: dict) -> np.ndarray:
    # TODO: Convert one document's token list into a bag-of-words count vector...
    vocab_size=len(vocab)

    bow =np.zeros(vocab_size,dtype=float)
    for token in tokens:
        if token in vocab:
            index=vocab[token]
            bow[index]+=1
    return bow

# Step 8 - corpus_to_bow_matrix
import numpy as np


def corpus_to_bow_matrix(tokenized_docs: list, vocab: dict) -> np.ndarray:
    # Number of documents = number of rows
    n_docs = len(tokenized_docs)

    # Number of vocabulary words = number of columns
    vocab_size = len(vocab)

    # Create an empty matrix filled with zeros
    # Shape = (number of documents, vocabulary size)
    # Example: 3 documents and 2 vocabulary words -> (3, 2)
    bow_matrix = np.zeros((n_docs, vocab_size), dtype=float)

    # Process every document one by one
    for i, tokens in enumerate(tokenized_docs):

        # Convert the current document into a 1-D BoW vector
        # Example: ['hello', 'hello'] -> [2., 0.]
        bow_vector = tokens_to_bow(tokens, vocab)

        # Put this document's BoW vector into row i
        bow_matrix[i] = bow_vector

    # Return the complete 2-D BoW matrix
    return bow_matrix

# Step 9 - compute_document_frequencies
import numpy as np


def compute_document_frequencies(bow_matrix: np.ndarray) -> np.ndarray:
    # Check whether each word appears in each document
    # > 0 means word appears at least once
    # Example:
    # [[1, 0, 2],
    #  [0, 0, 1]]
    #
    # becomes:
    # [[True, False, True],
    #  [False, False, True]]
    appears = bow_matrix > 0

    # Count True values column-wise
    # axis=0 means "look down each column"
    # Each True represents one document containing that word
    df = np.sum(appears, axis=0)

    # Return document frequency for every vocabulary word
    return df

# Step 10 - compute_idf
def compute_idf(df: np.ndarray, n_docs: int) -> np.ndarray:
    # TODO: Compute smoothed IDF idf_j = log((n_docs + 1) / (df_j + 1)) + 1
    numerator= n_docs+1
    denominator=df+1
    idf= np.log(numerator/denominator)+1
    return idf 

    #IDF tells us how rare a word is across documents.

# Step 11 - transform_tfidf (not yet solved)
# TODO: implement

# Step 12 - fit_tfidf (not yet solved)
# TODO: implement

# Step 13 - sigmoid (not yet solved)
# TODO: implement

# Step 14 - logistic_predict_proba (not yet solved)
# TODO: implement

# Step 15 - binary_cross_entropy (not yet solved)
# TODO: implement

# Step 16 - logistic_gradients (not yet solved)
# TODO: implement

# Step 17 - initialize_logistic_params (not yet solved)
# TODO: implement

# Step 18 - gradient_descent_step (not yet solved)
# TODO: implement

# Step 19 - train_logistic_regression (not yet solved)
# TODO: implement

# Step 20 - predict_labels (not yet solved)
# TODO: implement

# Step 21 - confusion_counts (not yet solved)
# TODO: implement

# Step 22 - metrics_from_counts (not yet solved)
# TODO: implement

# Step 23 - tune_decision_threshold (not yet solved)
# TODO: implement

# Step 24 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

