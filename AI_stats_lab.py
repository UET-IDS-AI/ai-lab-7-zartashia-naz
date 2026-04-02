"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    """
    Implement Naive Bayes spam classification using simple MLE.

    Returns
    -------
    priors : dict
    word_probs : dict
    prediction : int
    """
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # Step 1: Tokenize the texts
    tokenized = [text.lower().split() for text in texts]

    # Step 2: Build vocabulary (unique words across all documents)
    vocab = set(word for doc in tokenized for word in doc)

    # Step 3: Compute class priors  P(class)
    n_total = len(labels)
    classes = np.unique(labels)
    priors = {}
    for c in classes:
        priors[c] = float(np.sum(labels == c) / n_total)

    # Step 4: Compute word probabilities using simple MLE (no smoothing)
    # P(word | class) = count(word in class docs) / total words in class docs
    word_probs = {}
    for c in classes:
        # Collect all words in documents belonging to class c
        class_words = []
        for doc, label in zip(tokenized, labels):
            if label == c:
                class_words.extend(doc)

        total_word_count = len(class_words)

        word_probs[c] = {}
        for word in vocab:
            count = class_words.count(word)
            # MLE: probability is 0 if word never appears in this class
            word_probs[c][word] = count / total_word_count if total_word_count > 0 else 0.0

    # Step 5: Predict the class of test_email
    test_tokens = test_email.lower().split()

    best_class = None
    best_log_prob = float('-inf')

    for c in classes:
        # Start with log prior
        log_prob = np.log(priors[c])

        for word in test_tokens:
            if word in word_probs[c]:
                p = word_probs[c][word]
                if p == 0:
                    # Zero probability word → this class is impossible
                    log_prob = float('-inf')
                    break
                log_prob += np.log(p)
            # If word not in vocab at all, it contributes nothing (skip)

        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_class = c

    prediction = int(best_class)

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):
    """
    Implement KNN from scratch on the Iris dataset.

    Returns
    -------
    train_accuracy : float
    test_accuracy : float
    predictions : np.ndarray
    """
    # Step 1: Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Step 2: Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Step 3 & 4: Euclidean distance + KNN prediction
    def predict_knn(X_train, y_train, X_query, k):
        predictions = []
        for query_point in X_query:
            # Compute Euclidean distance from query_point to all training points
            distances = np.sqrt(np.sum((X_train - query_point) ** 2, axis=1))

            # Find k nearest neighbor indices
            k_nearest_indices = np.argsort(distances)[:k]

            # Majority vote among k nearest neighbors
            k_nearest_labels = y_train[k_nearest_indices]
            counts = np.bincount(k_nearest_labels)
            predicted_label = np.argmax(counts)
            predictions.append(predicted_label)

        return np.array(predictions)

    # Step 5: Compute train and test predictions
    train_predictions = predict_knn(X_train, y_train, X_train, k)
    test_predictions = predict_knn(X_train, y_train, X_test, k)

    # Step 6: Compute accuracies
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    predictions = test_predictions

    return train_accuracy, test_accuracy, predictions
