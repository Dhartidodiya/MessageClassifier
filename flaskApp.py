from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
from text_processing import clean_text, segment_text  # Importing from the pre-existing Python file

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and vectorizer
rf_model = joblib.load('random_forest_model.pkl')  # Random Forest model
vectorizer = joblib.load('tfidf_vectorizer.pkl')   # TF-IDF Vectorizer
label_encoder = joblib.load('label_encoder.pkl')   # Label Encoder for target labels

# Function to classify the message (clean, segment, vectorize, predict)
def classify_message(message):
    # Clean the message using the imported method
    cleaned_message = clean_text(message)
    
    # Segment the message into sentences using the imported method
    sentences = segment_text(cleaned_message)
    print(f"Segmented Sentences: {sentences}")  # Debugging line to see segmented sentences
    
    # If no sentences are segmented, return an error message
    if not sentences:
        return {"error": "No valid sentences found."}
    
    # Vectorize each sentence
    sentence_vectors = vectorizer.transform(sentences)
    
    # Predict the category for each sentence
    predictions = rf_model.predict(sentence_vectors)
    predicted_categories = label_encoder.inverse_transform(predictions)
    
    # Prepare the result as a list of sentences and their predicted categories
    result = [{'sentence': sentence, 'category': category} for sentence, category in zip(sentences, predicted_categories)]
    
    return result

# Define a route to handle POST requests for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']

    # Get the prediction results using the imported methods
    result = classify_message(message)
    
    # Return the result as a JSON response
    return jsonify(result)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
