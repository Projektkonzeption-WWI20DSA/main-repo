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
from word_2_vec_preprocessing import Preprocessing, Embedder, Clustering, Word2VecSummarizer

def load_model():
    corpus = api.load('text8')
    return Word2VecSummarizer(Word2Vec(corpus))
