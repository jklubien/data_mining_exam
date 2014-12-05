# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:45:59 2014

@author: Klubien
"""
import pickle
import nltk
import os
import matplotlib.pyplot as plt
import random
from nltk.corpus import names
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from nltk import FreqDist

"""
This method will open all books we have from the Gutenberg collection
(app. 500) and load them into a list.
"""


def open_all_books():
    if not os.path.isfile('book_list.txt'):
        book_list = []
        path = 'C:/Users/Klubien/AppData/Roaming/nltk_data/corpora/gutenberg'
        for file in os.listdir(path):
            if file.endswith('.txt'):
                book_list.append(str(file))
        with open('book_list.txt', 'wb') as handle:
            pickle.dump(book_list, handle)

    with open('book_list.txt', 'rb') as handle:
        book_list = pickle.load(handle)
    return book_list

"""
This methodwill only take books written by shakespeare and load them into
a list.
"""


def open_shakespeare_books():
    if not os.path.isfile('shakespeare_book_list.txt'):
        book_list = []
        path = 'C:/Users/Klubien/AppData/Roaming/nltk_data/corpora/gutenberg'
        for file in os.listdir(path):
            if file.startswith('sp'):
                book_list.append(str(file))
        with open('shakespeare_book_list.txt', 'wb') as handle:
            pickle.dump(book_list, handle)

    with open('shakespeare_book_list.txt', 'rb') as handle:
        book_list = pickle.load(handle)
    return book_list

"""
This method defines important features to identify the gender of a name
and returns them to the gender_features function
"""


def gender_features(word):
    features = {}
    features['last_letter'] = word[-1].lower()
    features['last_two_letters'] = word[-2:].lower()
    features['last_is_vowel'] = (word[-1].lower() in 'aeiouy')
    features['first_two_letters'] = word[:2].lower()
    return features


"""
Natural Language Processing with Python (O'Reilly 2009)
This method takes a name and defines the gender of it based on a training
set that has been tested against a test set
"""


def define_gender(name_input):
    """
    We only create a new classifier if we do not already have one saved.
    """
    if not os.path.isfile('train_set.txt') and not os.path.isfile('test_set'):
        """
        We take a sample of male and female names and mix
        them in order to create a training set and testing set
        """
        labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                         [(name, 'female') for name in names.words(
                             'female.txt')])
        random.shuffle(labeled_names)

        """
        We train the classifier and return the gender of the name
        """
        featuresets = [(gender_features(n), gender) for (n, gender)
                       in labeled_names]
        train_set, test_set = featuresets[-500:], featuresets[:500]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        with open('train_set.txt', 'wb') as handle:
            pickle.dump(train_set, handle)
        with open('test_set.txt', 'wb') as handle:
            pickle.dump(test_set, handle)
        with open('classifier.txt', 'wb') as handle:
            pickle.dump(classifier, handle)

    with open('train_set.txt', 'rb') as handle:
        train_set = pickle.load(handle)
    with open('test_set.txt', 'rb') as handle:
        test_set = pickle.load(handle)
    with open('classifier.txt', 'rb') as handle:
        classifier = pickle.load(handle)

    classifier = nltk.NaiveBayesClassifier.train(train_set)
#    accuracy = nltk.classify.accuracy(classifier, test_set)
#    classifier.show_most_informative_features(10)
#    print accuracy

    """
    Accuracy: .804
    Most Informative Features
             last_letter = u'a'           female : male   =     44.0 : 1.0
             last_letter = u'd'             male : female =     23.7 : 1.0
        last_two_letters = u'on'            male : female =     11.0 : 1.0
       first_two_letters = u'ha'            male : female =      7.8 : 1.0
        last_two_letters = u'ta'          female : male   =      7.0 : 1.0
             last_letter = u't'             male : female =      6.7 : 1.0
             last_letter = u'o'             male : female =      6.0 : 1.0
        last_two_letters = u'll'            male : female =      4.7 : 1.0
       first_two_letters = u'te'            male : female =      4.7 : 1.0
        last_two_letters = u'an'            male : female =      4.1 : 1.0
    """

    return classifier.classify(gender_features(name_input))

"""
This method takes a text as input and hopefully returns the author of the
text if written in English.
"""

def author_name(text):

    """
    We split the words so we can analyze them seperately
    """
    tag = text.split()

    """
    We take the beginning of the text since the
    author name will likely be there
    """

    tag = tag[:100]
    author = []

    current_tag = 0
    """
    We go through each word until we find the first instance
    of the word 'by' or 'author', which should mean the author
    will be written right after that.
    We save the first word after 'by' or 'author' since it should
    be the authors first name
    """

    for word in tag:
        if (word.lower() == ('by') or
                word.lower() == ('author') or
                word.lower() == ('author:')):

            author.append(tag[current_tag+1].decode(encoding='UTF8',
                          errors='ignore'))
            current_tag += 1
            tag = tag[current_tag+1:]
            break
        current_tag += 1

    """
    We go through each word after the first name of the author
    until we find a word that is not capitalized. We assume that
    it marks the end of the author name.
    We then return a list of the author's name split up.
    """
    current_tag = 0
    for word in tag:
        if tag[current_tag].lower() == 'this':
            break
        if tag[current_tag].istitle():
            author.append(tag[current_tag].decode(encoding='UTF8',
                          errors='ignore'))
            current_tag += 1

    return author

"""
This function checks if a given input is a number and
 is used in the publication_year() method.
"""


def is_number(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

"""
This method splits up the words of the text so we
get them seperately. For each word we check if the
first character is a number. If it is we check if the
second character is a number and so on.
If four characters in a row are numbers we return them
as the publication year. This will clearly only work for
books written before the year 1000.
"""


def publication_year(text):
    tag = text.split()
    for word in tag:
        if len(word) > 3:
            if (is_number(word[0]) and is_number(word[1]) and
                    is_number(word[2]) and is_number(word[3])):
                year = word[0] + word[1] + word[2] + word[3]
                year = int(year)
                if year < 2000:
                    return year


"""
http://blog.alejandronolla.com/2013/05/15/detecting-text-
language-with-python-and-nltk/
This method samples a text and returns the most likely language it is
written in
"""


def language_propability(text):

    lang_ratio = {}

    """
    We tokenize the words so we can analyze them seperately
    We take a sample of the text (word 2000 to word 2500).
    """
    tokens = wordpunct_tokenize(text)
    tokens = tokens[2000:2500]
    """
    We make them lower case to avoid capitalization errors
    """
    words = [word.lower() for word in tokens]

    """
    We go through each of the stopwords for the languages found
    in stopwords.fileids()
    We see how many of these words can be found in the text we
    have given and give it a score that we store for later use.
    If we have 8 instances of stopwords in the text we have given
    the score will be 8.
    """
    if not os.path.isfile('lang_stopw_list.txt'):
        lang_dict_list = []
        for lang in stopwords.fileids():
            temp_dict = {lang: stopwords.words(lang)}
            lang_dict_list.append(temp_dict)

        with open('lang_stopw_list.txt', 'wb') as handle:
            pickle.dump(lang_dict_list, handle)

    with open('lang_stopw_list.txt', 'rb') as handle:
        stopw = pickle.load(handle)

    for item in stopw:
        stopwords_set = item.values()
        stopwords_set = set(stopwords_set[0])
        words_set = set(words)
        intersection = words_set.intersection(stopwords_set)
        lang_ratio[str(item.keys())] = len(intersection)

    """
    Since we have a list of languages with different scores we simply
    take the language with the highest score and returns that.
    """
    prob_lang = max(lang_ratio, key=lang_ratio.get)

    return prob_lang


def get_freq_dist(text):

    book = nltk.corpus.gutenberg.raw(text)
    tokens = wordpunct_tokenize(book)
    freq_dist = FreqDist(tokens)
    most_common = freq_dist.most_common(50)

    return most_common


"""
Extract features from all books from a list of books
"""


def extract_features(book_list):

    feature_list = []

    for book in book_list:
        current_book = nltk.corpus.gutenberg.raw(book)
        author = author_name(current_book)
        author_gender = define_gender(author[0])
        year = publication_year(current_book)
        lang = language_propability(current_book)

        book_data = {'Book': book, 'Author': author,
                     'Author gender': author_gender, 'Publication year': year,
                     'Book language': lang}
        feature_list.append(book_data)

    return feature_list

"""
This method accumulates lists in order to plot in in a graph
"""
def acumulate_list(list):

    acumulated_list = []
    for item in list:
        acumulated_list.append(list.count(item))

    return acumulated_list

"""
This method generates a graph of the distribution of male and female
authors according to publication year with a list of books as input
"""
def gender_avg_pub_year(book_list):

    features = extract_features(book_list)
    male_list = []
    female_list = []

    for feat in features:
        if feat['Publication year'] is not None:
            current_year = int(feat['Publication year'])
            if (feat['Author gender'] == 'female' and
                    feat['Publication year'] is not None):
                female_list.append(current_year)
            if (feat['Author gender'] == 'male' and
                    feat['Publication year'] is not None):
                male_list.append(current_year)

    male_list.sort()
    female_list.sort()
    acum_male = acumulate_list(male_list)
    acum_female = acumulate_list(female_list)

    male, = plt.plot(male_list, acum_male, color='b', linewidth=1)
    female, = plt.plot(female_list, acum_female, color='r', linewidth=1)
    plt.plot(female_list, acum_female, 'ro')
    plt.plot(male_list, acum_male, 'bo')
    plt.axvline(sum(male_list)/len(male_list), color='b')
    plt.axvline(sum(female_list)/len(female_list), color='r')
    plt.legend([male, female], ['Male', 'Female'], loc=2)
    plt.xlabel('Year')
    plt.ylabel('Amount of books')

"""
This method takes a list of books from a specific author and creates a 
histogram of their distribution according to publication year. 
"""


def shakespeare_data(book_list):

    features = extract_features(book_list)
    pub_year = []

    for feat in features:
        if feat['Publication year'] is not None:
            pub_year.append(int(feat['Publication year']))
    plt.hist(pub_year, bins=50)
    plt.xlabel('Year')
    plt.ylabel('Amount of books')
    plt.show()

book_list = open_shakespeare_books()
#for book in book_list:
#    current_book = nltk.corpus.gutenberg.raw(book)
#    print publication_year(current_book)
book_list = book_list[:40]
#features = extract_features(book_list)
#lang = features[0]['Book language']  
#lang = lang[0]
#print lang  '
#english_count = 0
#males_count = 0
#females_count = 0
#total_count = 0
#for feat in features:
#    total_count += 1
#    if 'english' in feat['Book language']:   
#        english_count += 1
#    if feat['Author gender'] == 'male':
#        males_count += 1
#    if feat['Author gender'] == 'female':
#        females_count += 1
#        
#print english_count
#print males_count
#print females_count
#print total_count
        
#book_list = open_all_books()
#gender_avg_pub_year(book_list)
#book_list = book_list[:40]
#for book in book_list:
#    current_book = nltk.corpus.gutenberg.raw(book)
#    print publication_year(current_book)
#book = book.split()
shakespeare_data(book_list)
#shakespeare_data(book_list)
        