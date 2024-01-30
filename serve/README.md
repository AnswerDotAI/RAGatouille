# RAGatouille API Server

RAGatouille API is a FastAPI application designed to provide an efficient and scalable way to perform operations such as document indexing, searching, reranking, and encoding with the RAGPretrainedModel. It leverages the power of FastAPI to offer high performance and easy-to-use RESTful endpoints.

## Features

- **Document Indexing and Management**: Easily index and manage documents with comprehensive endpoints for adding, updating, and deleting documents.
- **Search and Rerank Functionality**: Perform advanced search queries and rerank search results to meet specific requirements.
- **Document Encoding and Retrieval**: Encode documents for efficient storage and retrieval, and search through encoded documents.
- **RESTful API**: Clear and concise endpoints adhering to RESTful standards, making the API easy to consume.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv

### Installation

1. Clone the repository:

   ```bash
   https://github.com/bclavie/RAGatouille.git
   cd RAGatouille/serve
   ```
2. Set up a virtual environment (optional but recommended): <br>
   Windows
   ```bash
   python -m venv venv
   source venv\Scripts\activate
   ```
   Others
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy .env.example to .env:
   ```bash
   cp .env.example .env
   ```

### Running the Application
Start the FastAPI server with:

```bash
uvicorn app.main:app --reload
```
The --reload flag enables hot reloading during development.

### Accessing the API Documentation
Once the server is running, you can view the auto-generated Swagger UI documentation by navigating to:
```bash
http://127.0.0.1:8000/docs
```

## API Endpoints

- **POST /api/v1/search/:** Search documents.
- **POST /api/v1/index/:** Create or update an index with documents.
- **POST /api/v1/add_to_index/:** Add documents to an existing index.
- **DELETE /api/v1/delete_from_index/:** Delete documents from an index.
- **POST /api/v1/rerank/:** Rerank a set of documents based on a query.
- **POST /api/v1/encode/:** Encode documents.
- **POST /api/v1/search_encoded_docs/:** Search through encoded documents.
- **POST /api/v1/clear_encoded_docs/:** Clear all encoded documents.