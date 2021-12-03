import csv
import os
import random
import time

import glob as gl
from collections import defaultdict

from PIL import Image

from Projekt_palgiarism.data_prepare_final import txt_to_words_string

used_colors = set()
# generate unique color for every word
def random_color():
    while 1:
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        c = random.randint(0, 255)
        tmp = (a, b, c)
        if tmp not in used_colors:
            used_colors.update([tmp])
            return tmp


# To prepare data for CNN it needs to be in form of images
# reshape 1d array of words to matrix
# if array of words is not long enough fill with zeroes
def words_array_to_matrix(words, frm, to, rows, columns):
    # maybe use np.pad(words, (0, 3), 'constant', constant_values=('0'))
    numpy = [0 for i in range(to)]
    for i in range(len(numpy)):
        if i < len(words):
            numpy[i] = words[i]
        else:
            numpy[i] = '0'
    return np.array(numpy).reshape(rows, columns)


# To prepare data for CNN it needs to be in form of images
# From given word matrix and dictionary mapping every word to color ([R,G,B]) create memory representation of image
def word_matrix_to_color_matrix(words, char2index):
    frames = np.zeros((words.shape[0], words.shape[1], 3))
    for i in range(words.shape[0]):
        for j in range(words.shape[1]):
            frames[i, j, :] = char2index[words[i][j]]
    return frames


# find frequency for every word, running time cca 180 s
def find_and_write_to_file_frequency_table():
    start = time.time()
    suspicious_files = gl.glob(
        os.path.join("pan-plagiarism-corpus-2010", "suspicious-documents", "*", "*.preprocessed"))
    frequency_dictionary = defaultdict(int)
    for file in suspicious_files:
        # problems with \ufeff https://programmersought.com/article/94395688398/, nearly no impact on performance
        words = txt_to_words_string(file).encode('utf-8').decode('utf-8-sig').split()
        for word in words:
            frequency_dictionary[word] += 1

    with open("frequency_table3.csv", "w", newline='', encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        for key, val in frequency_dictionary.items():
            writer.writerow([key, val])

    end = time.time()
    print('frequency table duration:', end - start)
    return frequency_dictionary


# creates dictionary running time cca 5 s, colors are generated randomly, but each word has different color
def map_words_to_colors_write_to_file_dictionary():
    start = time.time()
    with open("frequency_table2.csv", encoding='utf-8-sig') as f:
        unique_words = [row.split(',')[0] for row in f]
    words_to_color = dict((i, random_color()) for i in unique_words)

    with open("words_to_color_dictionary.csv", "w", newline='', encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for key, val in words_to_color.items():
            writer.writerow([key, val[0], val[1], val[2]])

    end = time.time()
    print('dictionary creation duration:', end - start)
    return words_to_color


# running time 1313 s
def create_pictures():
    start = time.time()

    with open("words_to_color_dictionary.csv", encoding='utf-8-sig') as f:
        reader = csv.reader(f, delimiter=',')
        word_to_color = dict((row[0], (int(row[1]), int(row[2]), int(row[3]))) for row in reader)

    suspicious_files = gl.glob(
        os.path.join("pan-plagiarism-corpus-2010", "suspicious-documents", "*", "*.preprocessed"))
    for file in suspicious_files:
        words = txt_to_words_string(file).encode('utf-8').decode('utf-8-sig').split()
        words = words_array_to_matrix(words, frm=0, to=40000, rows=200, columns=200)
        frames = word_matrix_to_color_matrix(words, word_to_color)
        im = Image.fromarray(frames.astype('uint8')).convert('RGB')
        im.save('pan-plagiarism-corpus-2010-images/{}.png'.format(os.path.basename(file)))

    end = time.time()
    print('picture creation duration:', end - start)


import numpy as np

if __name__ == "__main__":
    # # run only once
    # find_and_write_to_file_frequency_table()
    # # run only once
    # map_words_to_colors_write_to_file_dictionary()
    # # run only once
    # create_pictures()
    print('uncomment')

    ############# counting english texts, every text is english
    # suspisious_files = gl.glob(
    #     os.path.join("pan-plagiarism-corpus-2010", "suspicious-documents", "part1", "*.xml"))
    # counter=0
    # for s_file in suspisious_files:
    #     if next(iter(get_lang_source_reference_from_xml(s_file)[0])) == 'en':
    #         counter+=1
    # print(counter)
