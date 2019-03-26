import numpy as np
import pickle
import os
import glob
import re

def get_vocab_size(vocab_file = 'aclImdb/imdb.vocab'):
    sum = 0
    with open(vocab_file, 'rb') as file:
        for line in file:
            sum += 1
    return sum

def fit(pos_pickle='train_pos.pickle', neg_pickle='train_neg.pickle', vocab_file = 'aclImdb/imdb.vocab'):
    with open(pos_pickle, 'rb') as file:
        pos_dict = pickle.load(file)

    with open(neg_pickle, 'rb') as file:
        neg_dict = pickle.load(file)

    pos_class_count = pos_dict.pop('class_count')
    neg_class_count = neg_dict.pop('class_count')
    pos_class_prior = np.log2(pos_class_count/(pos_class_count + neg_class_count))
    neg_class_prior = np.log2(neg_class_count/(pos_class_count + neg_class_count))

    smooth_pos_dict = {key: val+1 for key, val in pos_dict.items()}
    smooth_neg_dict = {key: val+1 for key, val in neg_dict.items()}

    vocab_size = get_vocab_size(vocab_file)
    pos_sum = sum(pos_dict.values()) + vocab_size
    neg_sum = sum(neg_dict.values()) + vocab_size

    prob_dict = {}
    prob_dict['neg'] = {k: np.log2(v/neg_sum)
                        for k, v in smooth_neg_dict.items()}
    prob_dict['pos'] = {k: np.log2(v/pos_sum)
                        for k, v in smooth_pos_dict.items()}
    prob_dict['neg']['class_prior'] = neg_class_prior
    prob_dict['pos']['class_prior'] = pos_class_prior
    prob_dict['neg']['smooth_value'] = np.log2(1/neg_sum)
    prob_dict['pos']['smooth_value'] = np.log2(1/pos_sum)

    with open('movie-review-BOW.NB', 'wb') as file:
        pickle.dump(prob_dict, file)

    return prob_dict


def predict(filename):
    word_count_dict = {}

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = str(line)
            line = line.lower()
            line = re.sub('<[^<]+?>', '', line)
            new_line = ''
            for i in line:
                if i not in '!"#$%&()*+,./:;<=>?@[]^_`{|}~':
                    new_line += i
            line = new_line
            # append each train file
            for word in line.split():
                try:
                    word_count_dict[word] += 1
                except KeyError:
                    word_count_dict[word] = 1

    conditional_positive = prob_dict['pos']['class_prior']
    conditional_negative = prob_dict['neg']['class_prior']

    for key in word_count_dict.keys():
        try:
            conditional_positive += prob_dict['pos'][key] * word_count_dict[key]
        except KeyError:
            conditional_positive += prob_dict['pos']['smooth_value']
        try:
            conditional_negative += prob_dict['neg'][key] * word_count_dict[key]
        except KeyError:
            conditional_negative += prob_dict['neg']['smooth_value']
    if conditional_negative > conditional_positive:
        return 'neg'
    else:
        return 'pos'

def score(folder = 'aclImdb/test'):
    pos_folder = folder + '/pos'
    neg_folder = folder + '/neg'

    number_right = 0
    number_wrong = 0
    wrong_file_dict = {'false neg': [], 'false pos': []}
    for file in glob.glob(os.path.join(pos_folder, '*.txt')):
        if predict(file) == 'pos':
            number_right += 1
        else:
            number_wrong += 1
            wrong_file_dict['false neg'].append(file)

    for file in glob.glob(os.path.join(neg_folder, '*.txt')):
        if predict(file) == 'neg':
            number_right += 1
        else:
            number_wrong += 1
            wrong_file_dict['false pos'].append(file)

    print(f'Number right: {number_right}. Number wrong: {number_wrong}. Accuracy: {number_right/(number_right+number_wrong)}')
    return wrong_file_dict


prob_dict = fit()
prob_dict['pos']['class_prior']
wrong_files = score()


# Investigate results and see if we find trends

first_ten = wrong_files['false pos'][:10]

for line in first_ten:
    with open(line) as file:
        for l in file:
            print()
            print(l)
            print(bayes_probs(line))

def bayes_probs(filename):

    word_count_dict = {}

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = str(line)
            line = line.lower()
            line = re.sub('<[^<]+?>', '', line)
            new_line = ''
            for i in line:
                if i not in '!"#$%&()*+,./:;<=>?@[]^_`{|}~':
                    new_line += i
            line = new_line
            # append each train file
            for word in line.split():
                try:
                    word_count_dict[word] += 1
                except KeyError:
                    word_count_dict[word] = 1

    conditional_positive = prob_dict['pos']['class_prior']
    conditional_negative = prob_dict['neg']['class_prior']
    print(word_count_dict)
    for key in word_count_dict.keys():
        try:
            conditional_positive += prob_dict['pos'][key] * word_count_dict[key]
        except KeyError:
            conditional_positive += prob_dict['pos']['smooth_value']
        try:
            conditional_negative += prob_dict['neg'][key] * word_count_dict[key]
        except KeyError:
            conditional_negative += prob_dict['neg']['smooth_value']
    print(conditional_negative, conditional_positive)


# Train additional model:
# Write a function that creates a new vocabulary that disregards the hyphenated words
# Change around fit function to take in the new voacb file and output new parameters

def new_vocab(vocab_file = 'aclImdb/imdb.vocab'):
    with open(vocab_file, 'rb') as file:
        for line in file:
