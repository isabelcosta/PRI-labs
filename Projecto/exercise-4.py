import math
import os

''' Functions '''

def calculate_phraseness(ngrams_language_model_bg, one_gram_language_model_bg):
    return calculate_sigma(ngrams_language_model_bg, one_gram_language_model_bg)

def calculate_informativeness(one_gram_language_model_fg, one_gram_language_model_bg):
    return calculate_sigma(one_gram_language_model_fg, one_gram_language_model_bg)

def calculate_sigma(pw, qw):
    return pw * math.log(pw / qw)

def calculate_score(phraseness, informativeness):
    return phraseness + informativeness

''' Program Execution '''

# Get relative path to documents
relativePath = os.path.dirname(os.path.abspath(__file__));

# Read text from foreground file
foregroundFile = open(relativePath + "input-ex-4.txt", 'r')
foregroundContent = foregroundFile.read()
foregroundFile.close()

# Read text from background files
datasetPath = relativePath + "\\documents\\";

fileList = []

# Get all documents in "documents" directory into fileList
# Import documents into fileList
for docName in os.listdir(datasetPath):
    if docName.endswith(".txt"):
        f = open(datasetPath + docName, 'r')
        fileList += [f.read()]
        f.close()

# Tokenize text

# Sum informativeness and phraseness for each document

# Rank of n-grams by combination of informativeness and phraseness
