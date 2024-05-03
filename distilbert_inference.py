from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the trained model and tokenizer
model_path = './saved_model'  # Adjust this if your model is saved elsewhere
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def predict_sentiment(text):
    # Tokenize the text and convert to input IDs
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)

    # Get predictions from model
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get the predicted class (0 or 1)
    predicted_class = torch.argmax(probs, dim=-1).numpy()

    # Return class and probability of each class
    return predicted_class[0], probs.numpy()[0]

# Example usage:
text = "fuck you"
predicted_class, probabilities = predict_sentiment(text)
print(f"Predicted class: {predicted_class} (0: Negative, 1: Positive)")
print(f"Probabilities: {probabilities}")
