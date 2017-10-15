import matplotlib.pyplot as plt
import gensim
import pickle

from gensim.models import CoherenceModel, LdaModel
from gensim.corpora import Dictionary
import pyLDAvis.gensim

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Configuration
DATA_CLEAN      = 'clean_data/'
YEARS_AVAILABLE = list(range(2008,2018))
TOKEN_NO_BELLOW_FREQ   = 15
TOKEN_NO_ABOVE_PERCENT = 0.5

texts_years = []
for year in YEARS_AVAILABLE:
    with (open("{}{}.pickle".format(DATA_CLEAN, year), "rb")) as openfile:
        while True:
            try:
                texts_years.append(pickle.load(openfile))
            except EOFError:
                break

texts = []
for year in texts_years:
    for abstract in year:
        texts.append(abstract)

len(texts)

bigram = gensim.models.Phrases(texts)

texts = [bigram[line] for line in texts]

# Create a dictionary representation of the documents.
dictionary = Dictionary(texts)

# Filter out words that occur less than TOKEN_NO_BELLOW_FREQ documents,
# or more than TOKEN_NO_ABOVE_PERCENT% of the documents.
dictionary.filter_extremes(no_below=TOKEN_NO_BELLOW_FREQ, no_above=TOKEN_NO_ABOVE_PERCENT)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(text) for text in texts]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())

    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()

    return lm_list, c_v

lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=corpus, texts=texts, limit=20)

# ldamodel = LdaModel(corpus=corpus, num_topics=8, id2word=dictionary)
# Set training parameters.
num_topics = 14
chunksize = 2000
passes = 20
iterations = 400
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

ldamodel = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every)

pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
p = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(p, 'graphs/save_14_LDA_html')
