import glob
import os
import re
import pickle


def pre_process(write_file, folder):

    train_files = glob.glob(os.path.join(folder, '*.txt'))
    word_count_dict = {'class_count': len(train_files)}

    with open(write_file, "wb") as outfile:
        for filename in train_files:
            # Clean all train files: remove special characters, make lowercase, make a string
            # Account for '-' between words, aka do not delete these
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
        pickle.dump(word_count_dict, outfile)


def main():
    pre_process(write_file="train_pos.pickle", folder = 'aclImdb/train/pos')
    pre_process(write_file="train_neg.pickle", folder = 'aclImdb/train/neg')

    pre_process(write_file = 'small_pos.pickle', folder = 'small_corpus/train/pos')
    pre_process(write_file = 'small_neg.pickle', folder = 'small_corpus/train/neg')

    with open('small_neg.pickle', 'rb') as file:
        test_dict = pickle.load(file)

    test_dict


main()
