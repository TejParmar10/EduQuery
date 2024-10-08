# EduQuery - AI-Powered PDF Question Answering

**EduQuery** is an AI-powered tool designed to simplify the learning process for students. Built using the **Phi-1.5 model** and **Streamlit**, EduQuery allows users to upload PDFs (e.g., study notes, textbooks) and ask questions based on the content of the documents. This project provides answers using **Conversational Retrieval-Augmented Generation (RAG)** to make studying faster, more efficient, and tailored to specific topics.

## Features

- **Upload PDFs**: Users can upload multiple PDFs related to any subject.
- **Ask Questions**: EduQuery retrieves content from the uploaded PDFs and provides answers based on the context.
- **Conversational Interface**: The chatbot interacts with users in a conversational manner.
- **Efficient Learning**: A time-saving tool for students who need quick answers from study materials.

## Tech Stack

- **Model**: Phi-1.5 (powered by Hugging Face Transformers)
- **Interface**: Streamlit for user interaction and visualization
- **Libraries**: LangChain for conversation, FAISS for vector storage, and PyPDF2 for handling PDFs

## Installation

To run EduQuery locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/eduquery.git
   cd eduquery
2. Create and activate a virtual environment (optional but recommended):

   ```bash
     python3 -m venv env
     source env/bin/activate  # On Windows use `env\Scripts\activate`
3. Install the dependencies:

   ```bash
     pip install -r requirements.txt
4. Run the Streamlit app:

   ```bash
     streamlit run app.py
This will launch the application in your browser at http://localhost:8501.
## How It Works
- **Upload PDFs** : Users upload their PDF files, which are then processed and chunked for easier retrieval.
- **Ask Questions** : Type your question in the chat interface, and EduQuery will search the uploaded documents to provide relevant answers.
- **Conversation Flow** : The chatbot maintains a conversational flow, remembering previous interactions for smoother conversations.
## License
This project is licensed under the MIT License. See the LICENSE file for more details.
