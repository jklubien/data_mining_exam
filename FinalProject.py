# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 13:45:59 2014

@author: Klubien
"""

import nltk, random
from nltk.corpus import names
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize

gutenberg_books = ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt', 'bible-kjv.txt', 'blake-poems.txt', 'bryant-stories.txt', 'burgess-busterbrown.txt', 'carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt', 'edgeworth-parents.txt', 'melville-moby_dick.txt', 'milton-paradise.txt', 'shakespeare-caesar.txt', 'shakespeare-hamlet.txt', 'shakespeare-macbeth.txt', 'whitman-leaves.txt']

#This method defines important features to identify the gender of a name and returns them to the gender_features function
def gender_features(word):
    features = {}
    features['last_letter'] = word[-1].lower()
    features['last_two_letters'] = word[-2:].lower()
    features['last_is_vowel'] = (word[-1].lower() in 'aeiouy')
    features['first_two_letters'] = word[:2].lower()
    return features

### Natural Language Processing with Python (O'Reilly 2009) ###
def define_gender(name_input):
    #We take a sample of male and female names and mix them in order to create a training set
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(labeled_names)
    
    #We train the classifier and return the gender of the name
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    train_set = featuresets[-500:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
 
    return classifier.classify(gender_features(name_input))

def author_name(text):
    
    #We take the beginning of the text since the author name will likely be there
    text = text[:500]
    #We split the words so we can analyze them seperately
    tag = text.split()
    author = []

    current_tag = 0
    #We go through each word until we find the first instance of the word 'by' or 'author', which should mean the author will be written right after that
    #we save the first word after 'by' or 'author' since it should be the authors first name
    for word in tag:        
        if word.lower() == ('by') or word.lower() == ('author') or word.lower() == ('author:'):
            author.append(tag[current_tag+1])
            current_tag+=1
            tag = tag[current_tag+1:]
            continue
        current_tag+=1
    
    #We go through each word after the first name of the author until we find a word that is not capitalized. We assume that it marks the end of the author name.
    #We then return a list of the author's name split up.
    current_tag = 0
    for word in tag:
        if tag[current_tag].istitle():
            author.append(tag[current_tag])
            current_tag+=1

    return author

#This function checks if a given input is a number and is used in the publication_year() method
def is_number(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

#This method splits up the words of the text so we get them seperately
#For each word we check if the first character is a letter. If it is we check if the second character is a letter and so on.
#If four characters in a row are letters we return them as the publication year. This will clearly only work for books written after the year 999.
def publication_year(text):
    tag = text.split()
    for word in tag:
        if is_number(word[0]):
            if is_number(word[1]):
                if is_number(word[2]):
                    if is_number(word[3]):
                        return word[0] + word[1] + word[2] + word[3]


## http://blog.alejandronolla.com/2013/05/15/detecting-text-language-with-python-and-nltk/ ##                      
def language_propability(text):
    #We take a sample of the text (first 500 characters)
    text = text[:500]
    lang_ratio = {}
    
    #We tokenize the words so we can analyze them seperately
    tokens = wordpunct_tokenize(text)
    #We make them lower case to avoid capitalization errors
    words = [word.lower() for word in tokens]
    
    #We go through each of the stopwords for the languages found in stopwords.fileids()
    #We see how many of these words can be found in the text we have given and give it a score that we store for later use
    #If we have 8 instances of stopwords in the text we have given the score will be 8    
    for lang in stopwords.fileids():
        stopwords_set = set(stopwords.words(lang))
        words_set = set(words)
        intersection = words_set.intersection(stopwords_set)
        lang_ratio[lang] = len(intersection)
    
    #Since we have a list of languages with different scores we simply take the language with the highest score and returns that    
    prob_lang = max(lang_ratio, key = lang_ratio.get)
    
    return prob_lang
    
        

### Extract features from all books in Gutenberg collection
    
book_list = []
for book in gutenberg_books:
    current_book = nltk.corpus.gutenberg.raw(book)
    author = author_name(current_book)
    author_gender = define_gender(author[0])
    year = publication_year(current_book)
    lang = language_propability(current_book)
    
    book_data = {'Book': book, 'Author': author, 'Author gender': author_gender, 'Publication year': year, 'Book language': lang}
    book_list.append(book_data)

for book in book_list  :  
    print book
    print
