# MyFlask ML Project

## Description

This project uses Flask as the backend and applies pretrained Machine Learning algorithms summarize and classify text. It is possible to type text into the input box, upload a file or use speech to text.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have a `Windows/Linux/Mac` machine.
* You have installed Pthyon version 3.8 or newer
* You have installed pip
* Optional: You have created a new python enviroment

## Installing MyFlask ML Project

To install MyFlask ML Project, follow these steps:

Linux and macOS:
```bash
# Clone the repository
git clone https://github.com/Projektkonzeption-WWI20DSA/main-repo

# Navigate into the directory
cd main-repo

# Install dependencies
pip install -r requirements.txt
```

Windows:
```cmd
# Clone the repository
git clone https://github.com/Projektkonzeption-WWI20DSA/main-repo

# Navigate into the directory
cd main-repo

# Install dependencies
pip install -r requirements.txt
```

## Using MyFlask ML Project

To use this Project, follow these steps:  

Download the classification model from   
save the model to `./models/`

Run the Flask Server:  
run `flask run`

Stop the Flask Server:  
Press `Cmd` + `C`

## File Structure
The file structure of the project looks like this:

```
main-repo/
|-- app.py
|-- models/
|   |-- classificaion.py
|   |-- load_model.py
|   |-- word_2_vec_preprocessing.py
|   |-- trained_classification_model.joblib.py
|-- templates/
|   |-- index.html
|-- static/
|   |-- css
|   |-- img
|   |   |-- icon.svg
|   |-- js
|-- requirements.txt
|-- README.md
|-- .gitignore
```

## User-Manual
The Website is split in the areas:  
1. On the Top you can enter the text. For this you can type a text into the input box, upload a file or use text to speech.  
   After that, choose the desired processing options. Then press `'process text'`.  
   ** File Upload: Note that you can only choose between .docx and .txt files.**  
   ** Voice Input: Note that this function does not work in Firefox **
3. The summary box will display the summary.
4. The classification box will give the predicted class and the percentages for all classes.

![screenshot](https://github.com/Projektkonzeption-WWI20DSA/main-repo/assets/105482809/2ed28662-d274-4508-8af3-da2a10f3aae1)



