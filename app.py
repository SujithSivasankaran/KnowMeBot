from flask import Flask, request, jsonify
import json
from dotenv import load_dotenv
import warnings
from chat import custom_qa

load_dotenv()

app = Flask(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Chat history storage
chat_history = []

# Main interaction loop
@app.route('/chat', methods=['GET'])
def try_chat():
    question = request.args.get('question')
    chat_history = json.loads(request.args.get('chat_history'))

    answer = custom_qa(question, chat_history)
    chat_history.append((question, answer))
    return jsonify({"answer": answer, "chat_history": chat_history})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
