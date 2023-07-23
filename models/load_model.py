import gensim.downloader as api
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
import re
from gensim.models.word2vec import Word2Vec
import string
from typing import Union
from models.word_2_vec_preprocessing import Preprocessing, Embedder, Clustering, Word2VecSummarizer

def load_model():
    corpus = api.load('text8')
    return Word2VecSummarizer(Word2Vec(corpus))

def count_phrases(text):
    def split_sentence(text):
        """used to split given text into sentences"""
        sentences = sent_tokenize(text)
        return [sent for sent in sentences]
    return len(sent_tokenize(text))

def summarize_text(text, len_text, compression_rate, summarizer):
    sentences = summarizer.summarize(text, num_sentences=(len_text-round(len_text*compression_rate)))
    summary = ''
    for i in sentences:
        summary = summary + ' ' + i
    return summary