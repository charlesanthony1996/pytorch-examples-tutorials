from transformers import MarianMTModel, MarianTokenizer
import torch

def translate_text(text, src_lang='en', tgt_lang='fr'):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'

    # Load the tokenizer and the model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt')

    # Generate the translation
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # Decode the generated tokens to text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translated_text

if __name__ == '__main__':
    text = "Hello, how are you?"
    translated_text = translate_text(text)
    print(f"Original text: {text}")
    print(f"Translated text: {translated_text}")
