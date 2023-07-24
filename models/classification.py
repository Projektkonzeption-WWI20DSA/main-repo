import joblib
import numpy as np

def classify_text(text):
    # Load the trained model
    loaded_model = joblib.load('./models/trained_classification_model.joblib')

    # Make prediction using the loaded model
    probabilities = loaded_model.predict_proba([text])[0]

    # Get the class labels from the loaded model
    classes = loaded_model.classes_

    # Get the predicted class label (highest probability)
    predicted_class_index = np.argmax(probabilities)
    predicted_class = classes[predicted_class_index]

    # Sort the probabilities and classes in descending order to get the ranking
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probabilities = probabilities[sorted_indices]
    sorted_classes = classes[sorted_indices]

    # Calculate the percentage of certainty for each class
    total_probability = np.sum(sorted_probabilities)
    percentages = (sorted_probabilities / total_probability) * 100

    # Create a dictionary with the class and its corresponding probability percentage
    class_percentage_dict = {class_label: percentage for class_label, percentage in zip(sorted_classes, percentages)}

    # Combine the predicted class with the ranking of classes
    result = {
        "predicted_class": predicted_class,
        "class_percentages": class_percentage_dict
    }

    # make the results more readable
    def format_result(result):
        res_str = f'Predicted class: {result["predicted_class"]}\n\n'
        res_str += 'Class percentages:\n'
        for class_label, percentage in result['class_percentages'].items():
            res_str += f'{class_label}: {percentage}\n'
        return res_str  # This line is necessary to return the string

    formatted_result = format_result(result)

    print('Predicted Classes: '+formatted_result)

    return formatted_result
