from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax

# Setup Flask application
app = Flask(__name__)
app.debug = True
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


# Load the model and tokenizer directly
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
loaded_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")


"""
def generate_response(input_text):
    print(input_text)
    # Here you would perform your text filtering logic
    # For now, let's just return the input text as it is
    return jsonify({"filtered_text": input_text})
"""


# Route to handle text filtering
@app.route('/api/test', methods=['POST'])
def filter_text():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Call the generate_response function to filter the text
        response = generate_response(text)
        return response

    except Exception as e:
        print("Error during text filtering:", str(e))
        return jsonify({"error": str(e)}), 500


# Real part uncommented:


# Define preprocess function
def preprocess(text):
    # You may need to customize this preprocess function based on how your model was trained
    return text

# Define a function to process each text
def process_text(text, threshold=0.3):
    # Preprocess the text
    text = preprocess(text)
    
    # Tokenize and encode the input text
    encoded_input = tokenizer(text, return_tensors='pt')
    
    # Feed input to the model
    output = loaded_model(**encoded_input)
    
    # Get scores
    scores = output.logits.detach().numpy()
    scores = softmax(scores)
    
    # Calculate the negativity score
    negativity_score = scores[0][0]  # Assuming binary classification
    
    # Print the text and negativity score
    print(f"All Texts: {text}")
    print(f"Negativity Score: {negativity_score:.4f}\n")
    
    # Check if the negativity score is higher than the threshold
    if negativity_score > threshold:
        return True, text, negativity_score
    else:
        return False, text, negativity_score

def generate_response(input_text):
    # Placeholder implementation, replace with actual logic
    
    
    #data = request.json
    #highlighted_text = data.get('text')
    
    highlighted_text = input_text
    
    #highlighted_text = "For starters bend over the one in pink and kick that ass and pussy to get a taste until she's begging for a dick inside her."

    # Process the text
    filtered, text, score = process_text(highlighted_text)
    if filtered:
        # Return the analyzed text and its negativity score
        return jsonify({"filtered_text": text})
    else:
        return jsonify({"filtered_text": 'Is not HS'})
    


# If running the Flask app directly
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7000)