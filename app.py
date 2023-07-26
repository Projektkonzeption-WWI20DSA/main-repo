from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import docx2txt
import os
from models import classification
import gensim.downloader as api
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
import re
from gensim.models.word2vec import Word2Vec
import string
from typing import Union
from models.word_2_vec_preprocessing import Preprocessing, Embedder, Clustering, Word2VecSummarizer
from models.load_model import load_model,summarize_text,count_phrases

nltk.download('punkt')
summarizer = load_model()
print('MODEL LOADED')


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
        
        print("text:\t", text)

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
                compression=request.form.get('compression-slider')
                if compression:
                    compression=compression
                    print("Compression Selected", compression)
                else:
                    compression = 0.5
                print("Summary Selected")
                #print('compression\t: ', compression)
                try:
                    compression = int(compression)/100
                    summary_result = summarize_text(text,count_phrases(text),compression,summarizer)
                    word_compression_rate =  len(word_tokenize(summary_result))/len(word_tokenize(text))
                    summary_result=summary_result + '\n'+'----------------------------'+'\n' + 'Erreichte Kompressionsrate:\t ' + str(round(word_compression_rate,2))
                except Exception as ex:
                    print(ex)
                    summary_result = "Sorry. Summarization failed."
                print('summary:\t', summary_result)
                content = text + "text wurde verarbeitet"

            if request.form.get('classification'):
                print("Classification Selected")
                
                try:
                    classification_result = classification.classify_text(text)
                except Exception as ex:
                    print(ex)
                    classification_result = "Sorry. Classification failed."

                content = text

            # processed_text = text

    # return render_template('index.html', content=content,processed_text=processed_text, summary_result=summary_result, classification_result=classification_result)
    return render_template('index.html', content=content, summary_result=summary_result, classification_result=classification_result) 

if __name__ == '__main__':
    app.run()
