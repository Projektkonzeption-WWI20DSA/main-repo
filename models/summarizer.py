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

#text='Im Kontext des Wissensmanagements steht "Interacting BA" für "Interaktive Wissensentwicklung". Dieser Begriff bezieht sich auf den Prozess der gemeinschaftlichen Erzeugung und Weiterentwicklung von Wissen durch Interaktion und Zusammenarbeit zwischen verschiedenen Akteuren.Bei der Interaktiven Wissensentwicklung spielen Mitarbeiter, Experten und andere Beteiligte eine aktive Rolle bei der Erstellung und Aufrechterhaltung des Wissens. Es geht darum, Wissen nicht nur als statische Ressource zu betrachten, sondern als etwas, das durch den Austausch von Erfahrungen, Ideen und Informationen entsteht und wächst.Interaktive Wissensentwicklung kann in verschiedenen Formen stattfinden, wie beispielsweise in Meetings, Workshops, Diskussionsforen, gemeinsamen Projekten oder Online-Kollaborationstools. Durch den regelmäßigen Austausch und die Zusammenarbeit wird neues Wissen geschaffen, bestehendes Wissen erweitert und aktualisiert sowie eine gemeinsame Wissensbasis aufgebaut.Diese Art des Wissensmanagements fördert den aktiven Dialog und die Beteiligung der Mitarbeiter und ermöglicht es einer Organisation, von der Expertise und dem Erfahrungsschatz ihrer Mitarbeiter zu profitieren. Durch die interaktive Wissensentwicklung können Innovationen gefördert, Problemlösungen gefunden und effiziente Arbeitsabläufe etabliert werden.text wurde verarbeitet'
#from models.load_model import load_model
#summarizer = load_model()
#print(summarize_text(text,count_phrases(text),0.5,summarizer))