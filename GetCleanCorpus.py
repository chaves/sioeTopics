from ftfy import fix_text # Fix any unicode problem
import spacy              # Prepare the data
import csv
import pickle
import glob, re


# Configuration
DATA_SOURCES = 'sources_data/'
GB_US_synonyms_file = DATA_SOURCES + 'gb-us-synonyms.txt'
expands_file = DATA_SOURCES + 'expands.txt'
DATA_CLEAN = 'clean_data/'
years_available = list(range(2008,2018))

# noisy_pos_tags = ['PROP'] # noisy tags
noisy_pos_tags = ["PROP","DET","PART","CCONJ","ADP","PRON","VERB","ADJ"]
# DET  = definite or indefinite article
# ADP  = conjunction, subordinating or preposition
# PART = adverb, particle
# ADP  = postposition => in
# PRON = pronoun, personal => I
# VERB, ADJ
min_token_length = 2 # minimum token length to remove
common_token = 20 # most characteristic keywords

# The first step to use spaCy is to constructs a
# language processing pipeline, here we're loading
# the pre-trained english model

nlp = spacy.load("en")

def is_noise(token):
    '''
    standard way to validate spacy tokens
    This method validate all the passed tokens and set true false on it
    '''
    is_noise = False
    if token.pos_ in noisy_pos_tags:
        is_noise = True
    elif token.is_stop == True:
        is_noise = True
    elif token.is_digit == True:
        is_noise = True
    elif token.is_punct == True:
        is_noise = True
    elif token.is_space == True:
        is_noise = True
    elif len(token.string) <= min_token_length:
        is_noise = True
    return is_noise

my_stop_words = ['effect', 'find', 'iii', 'e.g.', 'i.e.', 'al.', 'evidence', 'article', 'paper', \
                 'result', 'results', 'author', 'authors', 'v.s.']
for stop in my_stop_words:
    nlp.vocab[stop].is_stop = True

def get_list(file):
    with open(file, mode='r') as file:
        terms = csv.reader(file)
        return {rows[0]:rows[1] for rows in terms}

def gb_to_us(words):
    '''
    Replace British English with American English
    Important since it concerns an international conference
    e.g. both organisation and organization terms are used regularly
    source : https://github.com/7digital/synonym-list/
    '''
    gb_us = get_list(GB_US_synonyms_file)
    for key in gb_us:
        words = words.replace(key, gb_us[key])
    return words


def remove_specific_stop(words):
    punct = ['%', ',', '/', '(', ')', '.'] # frequent punctuation terms inside strings or digits
    for p in punct:
        words = words.replace(p, ' ')
        words = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", words)
    return words


texts, article = [], []
texts_txt = ''
for year in years_available:
    file_list = glob.glob("sources_data/{}/*.txt".format(year))
    for f in file_list:
        words = open(f).read()
        words = fix_text(words) # Fix any unicode problem
        words = words.replace('\n', ' ').replace('\r', '') # remove line breaks
        words = remove_specific_stop(words)
        words = gb_to_us(words)
        if(len(words.split()) >= 30): # Only abstracts with at least 30 words
            nlp_words = nlp(words)
            for word in nlp_words:
                if not is_noise(word):
                    article.append(word.lemma_)
            texts.append(article)
            texts_txt = texts_txt + ' '.join(article) + '\n'
            article = []

    with open("{}{}.pickle".format(DATA_CLEAN, year), "wb") as fp:
        pickle.dump(texts, fp)

    with open("{}{}.txt".format(DATA_CLEAN, year), "w") as fp:
        fp.write(texts_txt)

    texts_txt = ''
    texts = []
    print('Year ' + str(year))
