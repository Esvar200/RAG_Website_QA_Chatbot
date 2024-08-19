# RAG Website_QA_Chatbot

**RAG Website_QA_Chatbot** is a powerful tool designed to extract information from a user-provided website, store it in a Chroma DB, and answer queries using a Retrieval-Augmented Generation (RAG) approach with Google's Gemini API. The application is built with Streamlit, providing an easy-to-use interface for interacting with the chatbot.

## Key Features

- **Website Content Parsing**: Extracts and processes content from any website provided by the user using Langchain.
- **RAG-Based Query Answering**: Utilizes Google's Gemini API for a RAG approach to generate accurate and context-aware answers.
- **Chroma DB**: Efficiently stores and retrieves parsed content for quick response times.
- **Streamlit Integration**: Provides a user-friendly web interface to interact with the chatbot.

## Technologies Used

- **Langchain**: For parsing and processing web content.
- **Google's Gemini API**: Implements a RAG-based approach for answering user queries.
- **Chroma DB**: A powerful vector database for storing and retrieving parsed data.
- **Streamlit**: Creates a simple, interactive web interface for user interactions.

## How to Run the Project

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/RAG_Website_QA_Chatbot.git
    cd RAG_Website_QA_Chatbot
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Chatbot Backend**:
    ```bash
    python chatbot.py
    ```

4. **Run the Streamlit Interface**:
    ```bash
    python streamlit.py
    ```

5. **Access the Web Interface**:
   - After running `streamlit.py`, open your web browser and go to `http://localhost:8501` to start interacting with the chatbot.

## File Structure

- **`chatbot.py`**: Handles the backend logic, including parsing website content and storing it in Chroma DB.
- **`streamlit.py`**: Manages the front-end interface, allowing users to interact with the chatbot through a web page.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any feature enhancements or bug fixes.
---
