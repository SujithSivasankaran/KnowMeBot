import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import warnings


load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if not os.getenv("OPENAI_API_KEY") or not os.getenv("INDEX_NAME"):
    raise ValueError(
        "Missing required environment variables: OPENAI_API_KEY or INDEX_NAME"
    )

app = Flask(__name__)
CORS(app)  


open_api_key = os.getenv("OPENAI_API_KEY")
index_name = os.getenv("INDEX_NAME")


warnings.filterwarnings("ignore")


chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")


embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


custom_prompt = PromptTemplate(
    input_variables=["question", "chat_history", "context"],
    template=(
        "If greeted, respond warmly and introduce yourself as Sujith, not an AI assistant. "
        "Respond in the first person, as if you are Sujith. Use only the context provided below to answer the question. "
        "If the answer is not in the context, say, 'I don't know, but I will get back to you.' "
        "Do not use any external information.\n\n"
        "Chat History:\n{chat_history}\n"
        "Context:\n{context}\n"
        "Question: {question}\n"
        "Please be specific in your response."
    )
)


qa_chain = LLMChain(llm=chat, prompt=custom_prompt)


summarization_prompt = PromptTemplate(
    input_variables=["question", "response"],
    template=(
        "Summarize the following interaction concisely:\n"
        "Question: {question}\n"
        "Response: {response}\n"
        "Summary:"
    )
)


summarization_chain = LLMChain(llm=chat, prompt=summarization_prompt)



def extract_relevant_keywords(question):
   
    return question.split()



def custom_qa(question, chat_history, max_history_length=5):
    try:
        
        relevant_terms = extract_relevant_keywords(question)

        
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(" ".join(relevant_terms))
        context = "\n".join([doc.page_content for doc in docs])

       
        formatted_chat_history = "\n".join(
            f"Q: {q}\nA: {a}" for q, a in chat_history[-max_history_length:]
        ) if chat_history else ""

        
        response = qa_chain.run(question=question, chat_history=formatted_chat_history, context=context)
        return response

    except Exception as e:
        logger.error(f"Error in custom_qa: {e}")
        return "An error occurred while processing your request. Please try again later."


@app.route('/chat', methods=['GET'])
def chat():
    try:
      
        question = request.args.get('question')
        if not question:
            return jsonify({"error": "Question parameter is required"}), 400

        chat_history = json.loads(request.args.get('chat_history', '[]'))

        
        answer = custom_qa(question, chat_history)

        
        summary = summarization_chain.run(question=question, response=answer)

        
        chat_history.append((summary, ))

        return jsonify({"answer": answer, "chat_history": chat_history})

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
