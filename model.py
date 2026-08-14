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
# ── Step 005  count_word_frequencies ──
def count_word_frequencies(tokenized_docs: list) -> dict:
    """Count how many times each unique token appears across the whole corpus.

    Args:
        tokenized_docs: List of documents, where each document is a list of tokens
                         (e.g. output of tokenize_corpus).

    Returns:
        dict mapping token -> total occurrence count (int) across ALL documents.
    """
    word_counts = {}

    for doc in tokenized_docs:
        for token in doc:
            # dict.get(token, 0) returns 0 if token not seen yet, else current count
            word_counts[token] = word_counts.get(token, 0) + 1

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

# Step 11 - transform_tfidf
def transform_tfidf(bow_matrix: np.ndarray, idf: np.ndarray) -> np.ndarray:
    # TODO: Multiply BoW counts by the fitted IDF vector to produce TF-IDF features.
    tfidf=bow_matrix*idf
    return tfidf

# Step 12 - fit_tfidf
import numpy as np


def fit_tfidf(bow_train: np.ndarray) -> np.ndarray:
    # Step 1: Calculate Document Frequency (DF)
    # DF batata hai ki har word kitne documents mein present hai
    df = compute_document_frequencies(bow_train)

    # Step 2: Get the number of training documents
    n_docs = bow_train.shape[0]

    # Step 3: Convert DF into smoothed IDF
    # Existing compute_idf() function ka use kar rahe hain
    idf = compute_idf(df, n_docs)

    # Step 4: Return the fitted IDF vector
    return idf

# Step 13 - sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    # TODO: Map logits to probabilities with a numerically stable logistic sigmoid.
    z = np.asarray(z, dtype=float)

    result = np.empty_like(z)

    # For positive values: 1 / (1 + exp(-z))
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))

    # For negative values: exp(z) / (1 + exp(z))
    # This avoids overflow from exp(-z)
    negative = ~positive
    exp_z = np.exp(z[negative])
    result[negative] = exp_z / (1 + exp_z)

    return result

# Step 14 - logistic_predict_proba
def logistic_predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    # TODO: Return P(y=1|x) for each row via linear scores and sigmoid
    scores= X @ w + b
    return sigmoid(scores)

# Step 15 - binary_cross_entropy
def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> float:
    # TODO: Compute mean binary cross-entropy plus L2 penalty on the weights.
    # Clip probabilities to avoid log(0)
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)

    #binary cross entropy 
    bce = -np.mean(
        y_true * np.log(y_prob)
        + (1 - y_true) * np.log(1 - y_prob)
    )

    # L2 regularization
    l2_penalty = l2_lambda * np.sum(w ** 2) / 2

    # Total loss
    return float(bce + l2_penalty)

# Step 16 - logistic_gradients
def logistic_gradients(X: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, w: np.ndarray, l2_lambda: float) -> tuple:
    """Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.

    Args:
        X: Feature matrix of shape (N, D).
        y_true: Binary labels of shape (N,).
        y_prob: Predicted probabilities of shape (N,).
        w: Weight vector of shape (D,).
        l2_lambda: L2 regularization strength.

    Returns:
        Tuple (dw, db) with dw shape (D,) and db a float.
    """
    # TODO: Compute gradients of BCE+L2 w.r.t. weights and bias for one full batch.
    N = X.shape[0]

    # Error between prediction and actual label
    error = y_prob - y_true

    # Gradient w.r.t. weights
    dw = (X.T @ error) / N

    # Add L2 regularization
    dw += l2_lambda * w

    # Gradient w.r.t. bias
    db = np.mean(error)

    return dw, float(db)

# Step 17 - initialize_logistic_params
def initialize_logistic_params(n_features: int) -> tuple:
    # TODO: Return a zero weight vector of shape (n_features,) and bias 0.0
    w=np.zeros(n_features)

    #initialize python as bias float 
    b=0.0
    return w,b

# Step 18 - gradient_descent_step
def gradient_descent_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, l2_lambda: float) -> tuple:
    # TODO: Run one full-batch gradient descent update; return (w_new, b_new, loss).
    
    # 1. Predict probabilities
    y_prob= logistic_predict_proba(X,w,b)

    # 2. Calculate loss BEFORE updating parameters
    loss = binary_cross_entropy(y, y_prob, w, l2_lambda)

    # 3. Calculate gradients
    dw, db = logistic_gradients(
        X, y, y_prob, w, l2_lambda
    )

    # 4. Gradient Descent update
    w_new = w - lr * dw
    b_new = b - lr * db

    return w_new, b_new, loss

# Step 19 - train_logistic_regression
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float, l2_lambda: float, n_epochs: int) -> tuple:
    # TODO: Initialize params and run n_epochs of full-batch GD, recording loss..
    
    # Initialize weights and bias
    w, b = initialize_logistic_params(X.shape[1])

    # Store loss after every epoch
    losses = []


    # Training loop
    for _ in range(n_epochs):

        # One full-batch gradient descent step
        w, b, loss = gradient_descent_step(
            X, y, w, b, lr, l2_lambda
        )

        # Record loss
        losses.append(loss)


    return w, float(b), losses

# Step 20 - predict_labels
def predict_labels(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert predicted probabilities into hard binary labels.

    Args:
        proba: 1-D array of probabilities in [0, 1], shape (N,).
        threshold: Decision threshold; proba >= threshold maps to 1.

    Returns:
        Integer array of shape (N,) with values in {0, 1}.
    """
    # TODO: Convert probabilities to hard binary labels via the threshold...
    return (proba >= threshold).astype(int)

# Step 21 - confusion_counts
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    # TODO: Return the four confusion-matrix counts (tp, fp, tn, fn) as Python ints
    # True Positive: actual 1, predicted 1
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # False Positive: actual 0, predicted 1
    fp = np.sum((y_true == 0) & (y_pred == 1))

    # True Negative: actual 0, predicted 0
    tn = np.sum((y_true == 0) & (y_pred == 0))

    # False Negative: actual 1, predicted 0
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return int(tp), int(fp), int(tn), int(fn)

# Step 22 - metrics_from_counts
def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total != 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy)
    }

# Step 23 - tune_decision_threshold
def tune_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: np.ndarray = None
) -> tuple:
    # Use the exact required default grid
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    best_threshold = float(thresholds[0])
    best_f1 = -1.0

    for threshold in thresholds:
        # Convert probabilities to 0/1 predictions
        y_pred = predict_labels(proba, threshold)

        # Get confusion-matrix counts
        tp, fp, tn, fn = confusion_counts(y_true, y_pred)

        # Calculate F1
        denominator = 2 * tp + fp + fn

        if denominator == 0:
            f1 = 0.0
        else:
            f1 = (2 * tp) / denominator

        # Only use > so ties keep the first threshold
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return float(best_threshold), float(best_f1)

# Step 24 - evaluate_predictions (not yet solved)
# TODO: implement

# Step 25 - vectorize_texts (not yet solved)
# TODO: implement

# Step 26 - predict_text (not yet solved)
# TODO: implement

# Step 27 - collect_prediction_errors (not yet solved)
# TODO: implement

