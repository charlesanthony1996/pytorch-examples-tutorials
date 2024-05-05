from openai import OpenAI
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq  

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
load_dotenv()

# Use environment variables
open_api_key2 = os.getenv('OPEN_API_KEY')
client = None

# Example for Submodules Tutorial
# b = 2

# llm = "openAI"
llm = "groq"

def initialize_openai_client():
    global client
    if open_api_key2:
        client = OpenAI(api_key=open_api_key2)
    else:
        raise ValueError("OPEN_API_KEY environment variable is not set.")

def initialize_groq_client():
    global client
    # Set your Groq API key here
    # Add your API key here
   

def get_client():
    if llm == "openAI":
        if client is None:
            initialize_openai_client()
        return client
    elif llm == "groq":
        if client is None:
            initialize_groq_client()
        return client

@app.route('/api/analyze_hate_speech', methods=['POST'])
def analyze_hate_speech():
    try:
        API_KEY = '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'
        
        client = Groq(api_key=API_KEY)
        client = get_client()

        if llm == "openAI":
            system_message = """You are an AI trained to detect hate speech or any other kind of offensive language and respond with counter-speech. 
                          If no hate speech is detected,
                          respond with 'No hate speech detected.'"""

            data = request.json
            user_message = data.get('text', '')
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
            )

            analysis_result = response.choices[0].message.content.strip()
            print("analysis result: ", analysis_result)
            return jsonify({"analysis_result": analysis_result}), 200

        elif llm == "groq":
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "you are a helpful assistant. Answer as Jon Snow"},
                    {"role": "user", "content": "Explain the importance of low latency LLMs"},
                ],
                model="llama3-70b-8192",
                temperature=0.5,
                max_tokens=150,
                top_p=1,
                stop=None,
                stream=False,
            )

            print(chat_completion.choices[0].message.content)
            return jsonify({"analysis_result": chat_completion.choices[0].message.content}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6001)