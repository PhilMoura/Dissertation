# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 08:28:57 2021

@author: phil
"""

## This file contains Twitter language processing functions

# Create and generate word cloud images:
# create one for all words (with # symbol removed) and another with hash_tags
# code taken from VA_Lab08 (Text)
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objs as go
#import plotly.plotly as py
import chart_studio.plotly as py
import cufflinks
pd.options.display.max_columns = 30
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
#from sklearn.decomposition import TruncatedSVD
#from sklearn.decomposition import LatentDirichletAllocation
#from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
output_notebook()

def plot_wordcloud(df, description):
    wordcloud_all = WordCloud(max_font_size=50, min_font_size=8, max_words=150, background_color="white",
                     width=800, height=400, prefer_horizontal=0.99).generate(df)
    
    # Display the generated image:
    _ = plt.figure(figsize = (16, 8))
    _ = plt.imshow(wordcloud_all, interpolation='bilinear')
    _ = plt.axis("off")
    
    description_ = "\n ----------- " + description + " ---------- \n"
    print(description_)
    plt.show()

# I want to see how many unique words
# code inspired by DataScience_Lab08
from nltk import FreqDist

def unique_words(in_text):
    text_tokens = word_tokenize(in_text)
    fdist_example = FreqDist(text_tokens)
    word_freq = dict((text_tokens, freq) for text_tokens, freq in fdist_example.items())

    return word_freq

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_k_n_gram(corpus, k=None, n=None):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:k]

def get_all_n_gram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq

def plot_ngrams(df, col_name):

    common_words = get_top_n_words(df[col_name], 25)
    df1 = pd.DataFrame(common_words, columns = [col_name, 'count'])
    df1.groupby(col_name).sum()['count'].sort_values(ascending=False).iplot(
        kind='bar', yTitle='Count', linecolor='black', title='Top 25 words in clean message text')

    common_bigrams = get_top_k_n_gram(df[col_name], 25,2)
    df3 = pd.DataFrame(common_bigrams, columns = [col_name , 'count'])
    df3.groupby(col_name).sum()['count'].sort_values(ascending=False).iplot(
        kind='bar', yTitle='Count', linecolor='black', title='Top 25 bigrams in clean message text')

    common_trigrams = get_top_k_n_gram(df[col_name], 25,3)
    df5 = pd.DataFrame(common_trigrams, columns = [col_name , 'count'])
    df5.groupby(col_name).sum()['count'].sort_values(ascending=False).iplot(
        kind='bar', yTitle='Count', linecolor='black', title='Top 25 trigrams in clean message text')