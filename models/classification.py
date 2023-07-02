import joblib

def summarize_text(text):
    # Load the trained model
    pipeline_02 = joblib.load('trained_model.joblib')

    # Perform the necessary operations to classify the text
    # Replace this with your actual text classification logic
    classification = pipeline_02.predict([text])[0]  # Replace with the generated classification

    classification = f"Sample Category. (original text: {text})"
    return classification
