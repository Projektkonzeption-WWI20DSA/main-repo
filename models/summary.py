import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download()

def process_summary(text):
    # Perform the necessary operations to generate the summary
    # Replace this with your actual summary generation logic

    # Tokenize text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize sentences into words and filter out stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = [word_tokenize(sentence) for sentence in sentences]
    filtered_words = [
        [word.lower() for word in sentence if word.isalpha() and word.lower() not in stop_words]
        for sentence in word_tokens
    ]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [
        [lemmatizer.lemmatize(word) for word in sentence]
        for sentence in filtered_words
    ]
    
    # Flatten the list of words into a single list
    flattened_words = [word for sentence in lemmatized_words for word in sentence]
    
    return ("Flattend words: " + str(flattened_words))