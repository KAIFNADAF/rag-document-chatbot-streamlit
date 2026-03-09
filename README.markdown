# RAG-Based Chatbot with Streamlit

This project implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain and Streamlit. It allows users to upload files (e.g., PDF, TXT, CSV, DOCX) and query their content, or chat generally with an AI assistant without uploading files. The chatbot supports conversational history, robust document loading, and error handling for issues like missing Poppler dependencies for PDFs.

## Features
- **General Chat Mode**: Chat with an OpenAI `gpt-4o-mini` model immediately after entering an API key, without needing to upload files.
- **RAG Chat Mode**: Upload files, index them, and query their content using a retrieval-augmented approach.
- **Supported File Formats**: PDF, TXT, CSV, DOCX, Markdown, HTML, JSON, with fallbacks for unsupported formats.
- **Robust Error Handling**: Handles Poppler-related issues for PDFs and provides clear error messages.
- **Debugging Tools**: Displays document/chunk counts and retrieved context for troubleshooting.
- **Conversational Memory**: Maintains chat history across general and RAG modes for seamless interactions.
- **Browser-Based UI**: Built with Streamlit for an intuitive, web-based interface.

## Prerequisites
- **Python**: 3.8 or higher.
- **OpenAI API Key**: Required for LLM and embeddings. Sign up at [OpenAI](https://platform.openai.com/) and obtain an API key.
- **Poppler (for PDFs)**: Optional but recommended for PDF processing.
  - **Windows**: Download from [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases), extract, and add `bin` to PATH (e.g., `C:\poppler\bin`).
  - **macOS**: `brew install poppler`
  - **Linux**: `sudo apt update && sudo apt install poppler-utils`
  - Verify: `pdftotext -v`
- **System Requirements**: At least 4GB RAM for small datasets; more for large files or complex queries.

## Installation
1. Clone or download this repository.
2. Install Python dependencies:
   ```bash
   pip install streamlit langchain langchain-openai langchain-community langchain-text-splitters langchainhub faiss-cpu openai pypdf docx2txt unstructured
   pip install "unstructured[pdf]"
   ```
3. (Optional) Install Poppler for PDF support (see Prerequisites).
4. Save the script as `app.py`.

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the provided URL in your browser (e.g., `http://localhost:8501`).
3. Enter your OpenAI API key in the sidebar.
4. **General Chat**:
   - Start chatting immediately by typing questions in the chat input (e.g., "What is RAG?").
   - The AI responds using `gpt-4o-mini` with conversational history.
5. **RAG Chat**:
   - Upload files (e.g., TXT, DOCX, PDF) via the file uploader.
   - Click "Index Uploaded Files" to process and index the documents.
   - Ask questions about the document content (e.g., "Summarize the document").
   - Check the "Retrieved Context" expander to debug retrieved chunks.
6. **Reset**: Click "Reset Chat History" to clear the conversation and indexed data.

## Example
- **General Chat**:
  - Input: "What is AI?"
  - Output: "AI is the simulation of human intelligence in machines, enabling tasks like learning and problem-solving."
- **RAG Chat**:
  - Upload a `test.txt` with: "This document discusses Retrieval-Augmented Generation (RAG)..."
  - Index the file.
  - Input: "What is RAG?"
  - Output: "RAG is Retrieval-Augmented Generation, a method combining retrieval and generation for better AI responses."

## Troubleshooting
- **"I don't know" Responses**:
  - Ensure documents are indexed (check document/chunk counts in the UI).
  - Verify query matches document content (use specific phrases or keywords).
  - Check "Retrieved Context" expander to see if relevant chunks were retrieved.
  - Try larger chunk sizes or more retrieved documents (edit `chunk_size` or `k` in the script).
- **Poppler Errors**:
  - If you see "Unable to get page count" or similar, install Poppler and ensure it’s in PATH.
  - Alternatively, use non-PDF files (e.g., TXT, DOCX) or enable `UnstructuredFileLoader`’s `fast` strategy.
  - Verify: `pdftotext -v`
- **Document Loading Errors**:
  - Check the error message in the UI.
  - Ensure files are not empty or corrupted.
  - Try a simple TXT file to test the pipeline.
- **API Key Issues**:
  - Ensure the OpenAI API key is valid and has sufficient quota.
  - Check for typos in the sidebar input.

## Project Structure
- `app.py`: Main script containing the Streamlit app, document loading, RAG pipeline, and chat logic.
- Dependencies: Managed via `pip` (see Installation).

## Future Enhancements
- **Local LLMs**: Integrate Hugging Face models for offline use.
- **Multimodal Support**: Add image or audio processing for advanced file types.
- **Advanced UI**: Enhance Streamlit with real-time file previews or query suggestions.
- **Vector Store Options**: Support Chroma or Pinecone for persistent storage.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Built with [LangChain](https://python.langchain.com/) for RAG and document processing.
- Powered by [Streamlit](https://streamlit.io/) for the web interface.
- Uses [OpenAI](https://openai.com/) for LLM and embeddings.
- Inspired by the need for a flexible, user-friendly RAG chatbot.