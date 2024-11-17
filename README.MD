# Personal Chatbot - KnowMeBot

This project implements a personal chatbot, **KnowMeBot**, designed to interact with users and answer questions based on personalized information provided in a document (PDF). The chatbot is deployed as a web service, allowing users to ask questions and receive answers about the developer (Sujith) as if they were interacting with the individual directly.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)


## Features

- **Personalized Responses**: The chatbot provides responses based on the data in a PDF document, imitating the developer's voice.
- **Conversational Memory**: The chatbot maintains a history of the conversation and uses it to provide contextually relevant answers.
- **Graceful Handling of Unknown Questions**: If a question is outside the scope of the provided data, the chatbot politely responds that it doesn’t know the answer but will get back to the user.
- **Web Integration**: It is designed to be integrated with a personal website, allowing visitors to interact with the chatbot directly on the site.

## Technologies Used

- **Python**: Programming language used to implement the chatbot.
- **LangChain**: A framework used to manage conversation and process documents for question-answering.
- **OpenAI GPT-3**: The language model used to generate responses.
- **Pinecone**: Vector database used to store and retrieve relevant data.
- **Flask**: Framework for creating the web service to interact with users.
- **PyPDF2**: Library used to extract text from the PDF document containing personalized information.

## How It Works

1. **Document Parsing**: The chatbot processes a PDF document (`about_me_data.pdf`) to extract information about the developer.
2. **Vectorization**: The extracted text is vectorized using OpenAI embeddings and stored in a Pinecone index for fast retrieval.
3. **User Interaction**: Users ask questions via a web interface. The chatbot retrieves the relevant context from the vector store and generates responses using the OpenAI model.
4. **Web Service**: The Flask app hosts the chatbot, providing a simple API for interaction.
