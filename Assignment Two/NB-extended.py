
"""Training additional model:"""

import numpy as np
import pickle
import os
import glob
import re
import nltk
from nltk.corpus import stopwords

# Write a function that creates a new vocabulary that drops stopwords
def new_vocab(vocab_file = 'aclImdb/imdb.vocab'):
    characters = ["!","?","(",")",":",";"]
    stoplist = stopwords.words('english') + characters
    new_vocab_list = []
    with open(vocab_file, 'rb') as file:
        for line in file:
            line = str(line)
            line = line[2:-3]
            line = ' '.join([word for word in line.split() if word not in stoplist])
            new_vocab_list.append(line)
    n=''
    for i in new_vocab_list:
        if(i==n):
            new_vocab_list.remove(i)

    with open('new_imdb.vocab', 'wb') as file:
        pickle.dump(new_vocab_list, file)

    return new_vocab_list

new_vocab()

with open('new_imdb.vocab', 'rb') as file:
    new_vocabulary = pickle.load(file)


# Running our get vocab size function from NB.py except with new vocab file
def get_vocab_size(vocab_file = 'new_imdb.vocab'):
    sum = 0
    with open(vocab_file, 'rb') as file:
        for line in file:
            sum += 1
    return sum

get_vocab_size()

# Running our fit function from NB.py except with new vocab file and pickling movie-review-extended.NB
def fit(pos_pickle='train_pos.pickle', neg_pickle='train_neg.pickle', vocab_file = 'new_imdb.vocab'):
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

    with open('movie-review-extended.NB', 'wb') as file:
        pickle.dump(prob_dict, file)

    return prob_dict

# Running our predict function from NB.py
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

# Running our score function from NB.py
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

# Results
prob_dict = fit()
prob_dict['pos']['class_prior']
wrong_files = score()
