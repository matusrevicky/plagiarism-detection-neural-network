import csv
import os
import string
from itertools import islice
import glob as gl


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# fast file reading
# returns one long string from text from given file (UTF-8 encoded)
def txt_to_words_string(filename):
    fin = open(filename, encoding='utf-8-sig')
    text = ''
    while True:
        next_n_lines = list(islice(fin, 1000000))
        if not next_n_lines:
            break
        text += " ".join(next_n_lines)
    return text.lower()


# natural language processing https://pythonspot.com/nltk-stop-words/ removes unnecessary words like ‘the’, ‘is’,
# ‘are’. This function also uses The Porter stemming algorithm (a process for removing the commoner morphological and
# inflexional endings from words in English.)
# !!! nltk module contains a list of stop words, which needs to be
# installed separately for installation guide refer to:
# https://pythonspot.com/tokenizing-words-and-sentences-with-nltk/
# return list of words
def text_preprocessing_to_word_array(data):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(data)
    words_filtered = []

    ps = PorterStemmer()
    translator = str.maketrans('', '', string.punctuation)
    for w in words:
        if w not in stop_words:
            tmp = ps.stem(w)
            tmp = tmp.translate((translator))
            if tmp:
                words_filtered.append(tmp)

    return words_filtered


if __name__ == "__main__":
    # !!! Estimated running time on cpu i7-4810MQ cca 2.5 hours, 15925 suspicious documents,
    # cca 622 219 312 words before processing
    # cca 300 000 000 words after processing
    ############ data processing start ###############
    # preprocess every suspicious document and save preprocessed results to file create
    # also save the amount of words after processing in a file
    suspicious_files = gl.glob(
        os.path.join("pan-plagiarism-corpus-2010", "suspicious-documents", "*", "*.txt"))
    dict_words_in_each_file = {}

    for file in suspicious_files:
        words = txt_to_words_string(file)
        x = words.split(" ")
        print(file, len(x))  # just to view progress
        words = text_preprocessing_to_word_array(words)
        print(file, len(words))  # just to view progress
        dict_words_in_each_file[file] = len(words)

        my_file = open("{}.preprocessed2".format(file), "w", encoding='UTF-8')
        my_list = map(lambda x: x + '\n', words)
        my_file.writelines(my_list)
        my_file.close()

    with open("preprocessed_files_amount_of_words2.csv", "w", newline='', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for key, val in dict_words_in_each_file.items():
            writer.writerow([key, val])

    ############ data processing end ###############
