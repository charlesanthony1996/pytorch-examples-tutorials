from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.special import softmax
import json

app = Flask(__name__)
app.debug = True
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

class BaseFilter:
    def filter_text(self, text):
        raise NotImplementedError("Method should be implemented by subclasses")

class TwitterRobertaFilter(BaseFilter):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    def filter_text(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = softmax(output.logits.detach().numpy())[0]
        return text, scores.argmax()

class BertFilter(BaseFilter):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    def filter_text(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = softmax(output.logits.detach().numpy())[0]
        return text, scores.argmax()

def load_filter():
    with open("config.json", "r") as f:
        config = json.load(f)
    filter_class = globals()[config["filter"]]
    return filter_class()

current_filter = load_filter()

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/api/test', methods=['POST'])
def filter_text():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        response = current_filter.filter_text(text)
        return jsonify({"text": response[0], "sentiment": response[1]})
    except Exception as e:
        print(f"Error during text filtering: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7001)
