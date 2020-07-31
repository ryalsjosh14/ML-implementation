import collections
import csv
import matplotlib.pyplot as plt
import numpy as np
import json

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    normalized_msg = message.lower()
    new_msg = normalized_msg.split()
    return new_msg

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """


    word_counts = {}
    word_dict = {}
    i = 0
    for message in messages:
        words = get_words(message)
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
            if word_counts[word] == 5:
                word_dict[word] = i
                i += 1
    return word_dict


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    transformed_text = np.zeros((len(messages), len(word_dictionary)))
    i = 0
    for message in messages:
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                transformed_text[i][word_dictionary[word]] += 1
        i += 1
    return transformed_text


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data (MxV)
        M: number of emails, V: number of unique words
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    #Compute probability that x = k given that message is ham and spam, respectively (add 1 for laplace smoothing)
    phi_k_ham = matrix[labels == 0].sum(axis=0) + 1
    phi_k_spam = matrix[labels == 1].sum(axis=0) + 1

    #Divide by total numner of words to get probability between 0 and 1 (add 2 for laplace smoothing)
    phi_k_ham /= (sum(phi_k_ham)+2)
    phi_k_spam /= (sum(phi_k_spam)+2)

    #Phi_k_spam.shape = phi_k_ham.shape = 1xV
    #Each element is the probability that a certain word is in the email given it is spam or ham, respectively

    phi_y = sum(labels) / len(labels)

    return phi_k_ham, phi_k_spam, phi_y


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: The trained model
    """

    #Get params from model
    phi_k_ham, phi_k_spam, phi_y = model

    #Find log-likelihood of spam and ham by multiplying matrix by transpose of phi_k
    spam_prob = (matrix.dot(np.log(phi_k_spam.T))) + np.log(phi_y)
    ham_prob = (matrix.dot(np.log(phi_k_ham.T))) + np.log(1-phi_y)
    #shapes of spam_prob and ham_prob = (M,1)

    #Compare vals of spam_prob and ham_prob. If spam_prob > ham_prob -> 1, else -> 0
    return np.greater(spam_prob, ham_prob).astype(int)


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # (2, V)
    phi_k_ham, phi_k_spam, phi_y = model
    ratio = np.log(phi_k_spam / phi_k_ham)
    #Sort the indices of the ratio array, and return the last 5 (greatest ratios)
    indices = np.argsort(ratio)[-5:]

    #Reverse the dictionary mapping words to indices
    rev_dict = {idx: word for word, idx in dictionary.items()}

    #Map the five indices to their respective words and append to solution array
    sol = []
    for index in indices:
        sol.append(rev_dict[index])

    return sol


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        eval_matrix: The word counts for the validation data
        eval_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    best_radius, best_acc = 0.0, 0.0

    
    for radius in radius_to_consider:
        pred = train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        print(pred)
        acc = (pred == val_labels).sum() / len(pred)
        if acc > best_acc:
            best_radius = radius
            best_acc = acc

    print(best_radius)
    return best_radius
    

def load_spam_dataset(tsv_path):
    """Load the spam dataset from a TSV file

    Args:
         csv_path: Path to TSV file containing dataset.

    Returns:
        messages: A list of string values containing the text of each message.
        labels: The binary labels (0 or 1) for each message. A 1 indicates spam.
    """

    messages = []
    labels = []

    with open(tsv_path, 'r', newline='', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for label, message in reader:
            messages.append(message)
            labels.append(1 if label == 'spam' else 0)

    return messages, np.array(labels)


def write_json(filename, value):
    """Write the provided value as JSON to the given filename"""
    with open(filename, 'w') as f:
        json.dump(value, f)


np.random.seed(123)


def train_and_predict_svm(train_matrix, train_labels, test_matrix, radius):
    """Train an SVM model and predict the resulting labels on a test set.

    Args:
        train_matrix: A numpy array containing the word counts for the train set
        train_labels: A numpy array containing the spam or not spam labels for the train set
        test_matrix: A numpy array containing the word counts for the test set
        radius: The RBF kernel radius to use for the SVM

    Return:
        The predicted labels for each message
    """
    model = svm_train(train_matrix, train_labels, radius)
    return svm_predict(model, test_matrix, radius)

def svm_train(matrix, category, radius):
    state = {}
    M, N = matrix.shape
    Y = 2 * category - 1
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(matrix.T)
    K = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (radius ** 2)) )

    alpha = np.zeros(M)
    alpha_avg = np.zeros(M)
    L = 1. / (64 * M)
    outer_loops = 10

    alpha_avg
    for ii in range(outer_loops * M):
        i = int(np.random.rand() * M)
        margin = Y[i] * np.dot(K[i, :], alpha)
        grad = M * L * K[:, i] * alpha[i]
        if (margin < 1):
            grad -=  Y[i] * K[:, i]
        alpha -=  grad / np.sqrt(ii + 1)
        alpha_avg += alpha

    alpha_avg /= (ii + 1) * M

    state['alpha'] = alpha
    state['alpha_avg'] = alpha_avg
    state['Xtrain'] = matrix
    state['Sqtrain'] = squared
    return state

def svm_predict(state, matrix, radius):
    M, N = matrix.shape
    output = np.zeros(M)
    Xtrain = state['Xtrain']
    Sqtrain = state['Sqtrain']
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(Xtrain.T)
    K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (radius ** 2)))
    alpha_avg = state['alpha_avg']
    preds = K.dot(alpha_avg)
    output = (1 + np.sign(preds)) // 2
    return output


def main():
    train_messages, train_labels = load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = load_spam_dataset('../data/ds6_test.tsv')
    
    dictionary = create_dictionary(train_messages)

    write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)
    #print(train_matrix)

    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    #print(naive_bayes_model)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    #print(naive_bayes_predictions)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    print(optimal_radius)

    write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))

if __name__ == "__main__":
    main()
