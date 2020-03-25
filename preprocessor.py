import numpy as np
import pandas as pd
from unidecode import unidecode

import re
import string

from nltk.tokenize import TweetTokenizer

import sys

# files of top 100 toxic words associated with each label
wordFiles = ["data/identity_hateWordFile.txt",
             "data/insultWordFile.txt",
             "data/threatWordFile.txt",
             "data/toxicWordFile.txt",
             "data/obsceneWordFile.txt",
             "data/severe_toxicWordFile.txt"]

toxic_words = set([])

length_threshold = 20000
word_count_threshold = 900
words_limit = 310000

valid_characters = " " + "@$" + "'!?-" + "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
valid_set = set(x for x in valid_characters)

astericks_words = [
                   ('mother****ers', 'motherfuckers'), ('motherf*cking', 'motherfucking'), ('mother****er', 'motherfucker'), ('motherf*cker', 'motherfucker'),
                   ('bullsh*t', 'bullshit'), ('f**cking', 'fucking'), ('f*ucking', 'fucking'), ('fu*cking', 'fucking'), ('****ing', 'fucking'),
                   ('a**hole', 'asshole'), ('assh*le', 'asshole'), ('f******', 'fucking'), ('f*****g', 'fucking'), ('f***ing', 'fucking'),
                   ('f**king', 'fucking'), ('f*cking', 'fucking'), ('fu**ing', 'fucking'), ('fu*king', 'fucking'), ('fuc*ers', 'fuckers'),
                   ('f*****', 'fucking'), ('f***ed', 'fucked'), ('f**ker', 'fucker'), ('f*cked', 'fucked'), ('f*cker', 'fucker'), ('f*ckin', 'fucking'),
                   ('fu*ker', 'fucker'), ('fuc**n', 'fucking'), ('ni**as', 'niggas'), ('b**ch', 'bitch'), ('b*tch', 'bitch'), ('c*unt', 'cunt'),
                   ('f**ks', 'fucks'), ('f*ing', 'fucking'), ('ni**a', 'nigga'), ('c*ck', 'cock'), ('c*nt', 'cunt'), ('cr*p', 'crap'), ('d*ck', 'dick'),
                   ('f***', 'fuck'), ('f**k', 'fuck'), ('f*ck', 'fuck'), ('fc*k', 'fuck'), ('fu**', 'fuck'), ('fu*k', 'fuck'), ('s***', 'shit'),
                   ('s**t', 'shit'), ('sh**', 'shit'), ('sh*t', 'shit'), ('tw*t', 'twat')]

# creates set of toxic words from all word_files
for link in wordFiles:
    word_file = open(link, "r")
    toxic_words = toxic_words.union(set(word_file.read().split("\n")))

# only include words with length greater than or equal to 4
toxic_words = set(filter(lambda x: len(x) >= 4, toxic_words))

# creates a set of printable characters
printable = set(string.printable)

def clean_text(x):
    """
    Function for cleaning text to remove characters, user identification, and
    non-printable characters that can interfere with the model's ability to make
    accurate predictions.
    """
    # remove newline characters
    x = re.sub('\\n',' ',x)

    # remove return characters
    x = re.sub('\\r',' ',x)

    # remove leading and trailing white space
    x = x.strip()

    # remove any text starting with User...
    x = re.sub("\[\[User.*", ' ', x)

    # remove IP addresses or user IDs
    x = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", ' ', x)

    # remove URLs
    x = re.sub("(http://.*?\s)|(http://.*)", ' ', x)

    # remove non_printable characters eg unicode
    x = "".join(list(filter(lambda c: c in printable, x)))

    return x

def split_word(word, toxic_words):
    """
    Function that accounts for toxic words being hidden within normal text. If
    any of the toxic words are found, it will split the non-toxic text, exposing
    the toxic word.
    """
    if word == "":
        return ""

    lower = word.lower()
    for toxic_word in toxic_words:
        lower = lower.replace(toxic_word, " " + toxic_word + " ")

    return lower

tknzr = TweetTokenizer(strip_handles=False, reduce_len=True)

def word_tokenize(sentence):
    """
    Function that tokenizes text and replaces associated characters with letters
    as well as adds spaces around punctuation.
    """
    sentence = sentence.replace("$", "s")
    sentence = sentence.replace("@", "a")
    sentence = sentence.replace("!", " ! ")
    sentence = sentence.replace("?", " ? ")

    return tknzr.tokenize(sentence)

def normalize_comment(comment):
    """
    Does a variety of normalizing steps, such as separating punctuation and
    decensoring, before splitting words within a sentence into a list of words.
    """
    comment = unidecode(comment)
    comment = comment[:length_threshold]

    normalized_words = []

    # replaces with associated word while maintaining uppercase, if needed.
    # example: W*rd becomes Word
    for w in astericks_words:
        if w[0] in comment:
            comment = comment.replace(w[0], w[1])
        if w[0].upper() in comment:
            comment = comment.replace(w[0].upper(), w[1].upper())

    # places spaces around .'s and ,'s if needed and exposes toxic words if hidden
    # example: bzt is a toxic word, 'Well bztuk you' becomes 'Well bzt uk you'
    for word in word_tokenize(comment):
        if word == "." or word == ",":
            normalized_words.append(word)
            continue

        if word.count(".") == 1:
            word = word.replace(".", " ")
        filtered_word = "".join([x for x in word if x in valid_set])

        normalized_word = split_word(filtered_word, toxic_words)

        normalized_words.append(normalized_word)

    normalized_comment = " ".join(normalized_words)

    result = []

    # maintains uppercase, if needed, and all else is lowercase
    for word in normalized_comment.split():
        if word.upper() == word:
            result.append(word)
        else:
            result.append(word.lower())

    return result
