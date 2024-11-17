from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
import os

open_api_key = os.getenv("OPENAI_API_KEY")
index_name = os.getenv("INDEX_NAME")

# Embedding model and vector store setup
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Chat model configuration
chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")

# Define a custom prompt template
custom_prompt = PromptTemplate(
    input_variables=["question", "chat_history", "context"],
    template=(
        "You are Sujith, not an AI assistant. Respond in the first person, as if you are Sujith. "
        "Use only the context provided below to answer the question. "
        "If the answer is not present in the context, say, 'I don't know, but I will get back to you.' "
        "Do not use any information outside the given context.\n\n"
        "Chat History:\n{chat_history}\n"
        "Context:\n{context}\n"
        "Question: {question}"
    )
)

# Create a custom LLM chain
qa_chain = LLMChain(llm=chat, prompt=custom_prompt)

def custom_qa(question, chat_history):
    try:
        # Retrieve relevant context from the vector store
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])

        # Format chat history
        formatted_chat_history = "\n".join(f"Q: {q}\nA: {a}" for q, a in chat_history) if chat_history else ""

        # Run the LLM chain
        response = qa_chain.run(question=question, chat_history=formatted_chat_history, context=context)
        return response
    except Exception as e:
        return f"An error occurred: {e}"
