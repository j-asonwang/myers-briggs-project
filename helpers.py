##################################################################
## File containing helper functions for MBTI classifier project ##
##################################################################

import itertools
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


# List of common emojis
##################################################################

emojis = [':)', ':]', ':|', ':[', ':(',
          ';)', ':/', ':o', ':O', ':D', ':&', ":')", ":'(", ':`(',
          'XD', ':X', '8X', 'K0', ':p', '=O', 'O:)', '>:)', '>:(', '%p',
          ':@)', '=^.^=', 'ó¿ò', 'ô¿ô', '°¿°', '©¿©', "'¿'", '?:)', '?:~)', '?8)',
          '8^)', 'B)', '=D', ':|', '=)', '=.=', '^_^', '^_', '^_~', '¬_¬',
          '>_<', 'o_o', 'O.O', 'o.O', "''", 'x_x', '\\m/><\\m/', 'ಠ_ಠ', ':*',
          '9_9', 'QQ', ':-)', ':-]', ':-|', ':-[', ':-(', ';-)', ':-/', ':-o',
          ':-O', ':-D', ':-&', ":'-)", ":'-(", ':`-(', 'X-D', ':-X', '8-X', 'K-0',
          ':-p', '=O', 'O:-)', '>:-)', '>:-(', '%-p', ':@)', '=^.^=', 'ó¿ò', 'ô¿ô',
          '°¿°', '©¿©', "'¿'", '?:-)', '?:~)', '?8-)', '8^)', 'B-)', '=D', ':|',
          '=)', '=.=', '^_^', '^_-', '^_~', '¬_¬', '>_<', 'o_o', 'O.O', 'o.O',
          "'-'", 'x_x', '\\m/><\\m/', 'ಠ_ಠ', ':*', '9_9', 'QQ', '@.@']

##################################################################


# Functions to engineer new features of raw text data
##################################################################

def count_caps(text):
    """Returns a count of capitalized letters in text"""
    counter = 0
    for letter in text:
        if letter.isupper():
            counter += 1
    return counter


def count_exclamations(text):
    """Returns a count of the number of ! in text"""
    counter = 0
    for letter in text:
        if letter == '!':
            counter += 1
    return counter


def count_digits(text):
    """Returns a count of the number of digits in text"""
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    counter = 0
    for letter in text:
        if letter in digits:
            counter += 1
    return counter


def count_emojis_shortcode(text):
    import re
    shortcodes = re.findall(r'\:[a-z]+\:', str(text))
    return len(shortcodes)


def count_emojis(text):
    emojis = [':)', ':]', ':|', ':[', ':(',
              ';)', ':/', ':o', ':O', ':D', ':&', ":')", ":'(", ':`(',
              'XD', ':X', '8X', 'K0', ':p', '=O', 'O:)', '>:)', '>:(', '%p',
              ':@)', '=^.^=', 'ó¿ò', 'ô¿ô', '°¿°', '©¿©', "'¿'", '?:)', '?:~)', '?8)',
              '8^)', 'B)', '=D', ':|', '=)', '=.=', '^_^', '^_', '^_~', '¬_¬',
              '>_<', 'o_o', 'O.O', 'o.O', "''", 'x_x', '\\m/><\\m/', 'ಠ_ಠ', ':*',
              '9_9', 'QQ', ':-)', ':-]', ':-|', ':-[', ':-(', ';-)', ':-/', ':-o',
              ':-O', ':-D', ':-&', ":'-)", ":'-(", ':`-(', 'X-D', ':-X', '8-X', 'K-0',
              ':-p', '=O', 'O:-)', '>:-)', '>:-(', '%-p', ':@)', '=^.^=', 'ó¿ò', 'ô¿ô',
              '°¿°', '©¿©', "'¿'", '?:-)', '?:~)', '?8-)', '8^)', 'B-)', '=D', ':|',
              '=)', '=.=', '^_^', '^_-', '^_~', '¬_¬', '>_<', 'o_o', 'O.O', 'o.O',
              "'-'", 'x_x', '\\m/><\\m/', 'ಠ_ಠ', ':*', '9_9', 'QQ', '@.@']
    counter = 0
    split = text.split(' ')
    for x in emojis:
        counter += sum(x in s for s in split)
    return counter

##################################################################


# Functions to return a confusion matrix
##################################################################

def show_cf(y_true, y_pred,
            class_names=None,
            model_name=None,
            labels=['INFP', 'ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ',
                    'ESFP', 'ESTJ', 'ESTP', 'INFJ', 'INTJ', 'INTP',
                    'ISFJ', 'ISFP', 'ISTJ', 'ISTP']):
    plt.figure(figsize=(10, 10))
    cf = confusion_matrix(y_true, y_pred)
    plt.imshow(cf, cmap=plt.cm.Blues)

    if model_name:
        plt.title("Confusion Matrix: {}".format(model_name))
    else:
        plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    class_names = labels
    tick_marks = np.arange(len(class_names))
    if class_names:
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

    thresh = cf.max() / 2.

    for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
        plt.text(j, i, cf[i, j], horizontalalignment='center',
                 color='white' if cf[i, j] > thresh else 'black')

    return plt.colorbar()

##################################################################


def show_roc_graph(model, X_train, X_test, y_train, y):
    y_bin = label_binarize(y, classes=['INFP', 'ENFJ', 'ENFP', 'ENTJ', 'ENTP',
                                       'ESFJ', 'ESFP', 'ESTJ', 'ESTP', 'INFJ',
                                       'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ',
                                       'ISTP'])
    classifier = OneVsRestClassifier(model)
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        colors = cycle(['blue', 'red', 'green'])
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()
