from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import docx2txt
import os
from models import summary, classification

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
    content = ''

    if request.method == 'POST':
        text = request.form.get('input-text')
        file = request.files.get('file')
        

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
                processed_text = text

            if request.form.get('classification'):
                print("Classification Selected")
                # classification_result = classification.process_classification(text)
                processed_text = text

            # processed_text = text

    # return render_template('index.html', content=content,processed_text=processed_text, summary_result=summary_result, classification_result=classification_result)
    return render_template('index.html', content=content) 

if __name__ == '__main__':
    app.run()
