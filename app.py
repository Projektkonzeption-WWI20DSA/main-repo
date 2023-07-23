from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import docx2txt
import os
from models import classification
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
from load_model import load_model

summarizer = load_model()
print('MODEL LOADED')

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

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = ["txt", "docx"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route('/', methods=['GET', 'POST'])
def index():
    processed_text = None
    summary_result = None
    classification_result = None
    content = 'Enter your text here or upload a file...'
   

    if request.method == 'POST':
        file = request.files.get('file')
        speech_text = request.form.get('speechText')

        # if speech_text != "":
        #     text = speech_text
        # else:
        text = request.form.get('input-text')
        
        print("text:\t", speech_text)

        if file and allowed_file(file.filename):
            if file.filename.endswith('.docx'):
                text = ''  # Leeren String initialisieren
                text += docx2txt.process(file)
                content = text
            elif file.filename.endswith('.txt'):
                text = ''  # Leeren String initialisieren
                text += file.read().decode('utf-8')
                content = text

            # Setzen Sie das Textfeld auf den Inhalt der hochgeladenen Datei
            # request.form['input-text'] = text

        if text:
            if request.form.get('summary'):
                print("Summary Selected")
                # summary_result = summary.process_summary(text)
                content = text + "text wurde verarbeitet"

            if request.form.get('classification'):
                print("Classification Selected")
                # classification_result = classification.process_classification(text)
                content = text

            # processed_text = text

    # return render_template('index.html', content=content,processed_text=processed_text, summary_result=summary_result, classification_result=classification_result)
    return render_template('index.html', content=content) 

if __name__ == '__main__':
    app.run()
