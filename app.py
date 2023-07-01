from flask import Flask, render_template, request, jsonify
from models import summary, classification  # Import the Python files
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/"

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/process-text', methods=['POST'])
# def process_text():
#     data = request.get_json()
#     text = data['text']

#     # Return the response
#     response = {'message': 'Text received and processed successfully'}
#     return jsonify(response)

@app.route('/process-summary', methods=['POST'])
def process_summary():
    data = request.get_json()
    text = data['text']
    
    # Process the summary using the 'summary.py' module
    summary_result = summary.process_summary(text)
    
    # Return the summary response
    response = {'summary': summary_result}
    return jsonify(response)

@app.route('/process-classification', methods=['POST'])
def process_classification():
    data = request.get_json()
    text = data['text']
    
    # Process the classification using the 'classification.py' module
    classification_result = classification.process_classification(text)
    
    # Return the classification response
    response = {'classification': classification_result}
    return jsonify(response)

# Uploading files

@app.route('/display', methods = ['GET', 'POST'])
def save_file():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        f.save(app.config['UPLOAD_FOLDER'] + filename)

        file = open(app.config['UPLOAD_FOLDER'] + filename,"r")
        content = file.read()
        
        
    return render_template('content.html', content=content) 

if __name__ == '__main__':
    app.run()
