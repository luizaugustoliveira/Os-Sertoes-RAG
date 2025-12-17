# Os Sertões - RAG System (Retrieval-Augmented Generation)

## About the Project

This project implements a **RAG (Retrieval-Augmented Generation)** system for analyzing and querying the classic Brazilian literary work "Os Sertões" by Euclides da Cunha. The system uses advanced natural language processing and information retrieval techniques to answer questions about the work in a contextualized and accurate manner.

## Objective

Create an intelligent assistant capable of answering questions about "Os Sertões" using three different RAG approaches:
- **Naive RAG**: Simple vector similarity retrieval
- **Parent Document Retriever**: Hierarchical document retrieval
- **Reranker RAG**: Retrieval with result reordering using Cohere

## System Architecture

```mermaid
graph TB
    subgraph "Data Input"
        PDF[os-sertoes.pdf]
    end

    subgraph "Document Processing"
        LOADER[PyPDFLoader<br/>Loads PDF]
        PDF --> LOADER
    end

    subgraph "Approach 1: Naive RAG"
        NAIVE_SPLIT[RecursiveCharacterTextSplitter<br/>chunks: 4000 chars<br/>overlap: 20]
        LOADER --> NAIVE_SPLIT
        
        VECTORSTORE_NAIVE[ChromaDB<br/>Vectorizes chunks]
        NAIVE_SPLIT --> VECTORSTORE_NAIVE
        
        SIMPLE_RET[Simple Retriever<br/>k=3 documents]
        VECTORSTORE_NAIVE --> SIMPLE_RET
    end

    subgraph "Approach 2: Parent Document RAG"
        PARENT_SPLIT[Parent Splitter<br/>chunks: 4000 chars<br/>overlap: 200]
        CHILD_SPLIT[Child Splitter<br/>chunks: 200 chars]
        LOADER --> PARENT_SPLIT
        PARENT_SPLIT --> CHILD_SPLIT
        
        DOCSTORE[InMemoryByteStore<br/>Stores parent documents]
        VECTORSTORE1[ChromaDB<br/>Vectorizes child chunks]
        
        CHILD_SPLIT --> VECTORSTORE1
        PARENT_SPLIT --> DOCSTORE
        
        PARENT_RET[ParentDocumentRetriever<br/>Retrieves full context]
        VECTORSTORE1 --> PARENT_RET
        DOCSTORE --> PARENT_RET
    end

    subgraph "Approach 3: Reranker RAG"
        TEXT_SPLIT[RecursiveCharacterTextSplitter<br/>chunks: 4000 chars<br/>overlap: 20]
        LOADER --> TEXT_SPLIT
        
        VECTORSTORE2[ChromaDB<br/>Vectorizes chunks]
        TEXT_SPLIT --> VECTORSTORE2
        
        NAIVE_RET[Naive Retriever<br/>k=10 documents]
        VECTORSTORE2 --> NAIVE_RET
        
        RERANK[CohereRerank<br/>rerank-v3.5<br/>top_n=3]
        NAIVE_RET --> RERANK
        
        COMP_RET[ContextualCompressionRetriever<br/>Compresses and reorders]
        RERANK --> COMP_RET
    end

    subgraph "Embeddings and LLM"
        EMB[OpenAI Embeddings<br/>text-embedding-3-small]
        LLM[ChatOpenAI<br/>gpt-3.5-turbo]
        
        EMB -.-> VECTORSTORE_NAIVE
        EMB -.-> VECTORSTORE1
        EMB -.-> VECTORSTORE2
    end

    subgraph "Response Generation"
        PROMPT[Prompt Template<br/>Brazilian literature expert]
        SIMPLE_RET --> PROMPT
        PARENT_RET --> PROMPT
        COMP_RET --> PROMPT
        
        PROMPT --> LLM
        LLM --> PARSER[StrOutputParser]
        PARSER --> ANSWER[Final Answer]
    end

    subgraph "Test Questions"
        Q1[Natural environment vision]
        Q2[Sertanejo population characteristics]
        Q3[Canudos War historical context]
        Q4[Antônio Conselheiro figure]
        Q5[Social and political criticism]
        
        Q1 --> PROMPT
        Q2 --> PROMPT
        Q3 --> PROMPT
        Q4 --> PROMPT
        Q5 --> PROMPT
    end

    style PDF fill:#e1f5ff
    style SIMPLE_RET fill:#ffe6e6
    style PARENT_RET fill:#fff4e6
    style COMP_RET fill:#f3e5f5
    style LLM fill:#e8f5e9
    style ANSWER fill:#fce4ec
```

## Approach Comparison

### Naive RAG (`naive_rag.ipynb`)

**Characteristics:**
- **Simple Chunking**: Divides documents into 4000-character chunks with 20-character overlap
- **Direct Retrieval**: Retrieves the 3 most similar documents based on vector similarity
- **Advantage**: Simple, fast, and easy to implement
- **Storage**: ChromaDB for vectors only

**Flow:**
1. Loads PDF and splits into chunks of 4000 characters
2. Stores chunks in ChromaDB
3. On query, retrieves the 3 most similar documents
4. Passes context directly to the LLM

### Parent Document RAG (`parent_rag.ipynb`)

**Characteristics:**
- **Hierarchical Chunking**: Divides documents into large chunks (parents) and small chunks (children)
- **Child Chunks**: 200 characters - used for vector search
- **Parent Chunks**: 4000 characters with 200-character overlap - returned as context
- **Advantage**: Precise search with broad context
- **Storage**: InMemoryByteStore for parent documents + ChromaDB for vectors

**Flow:**
1. Loads PDF and splits into pages
2. Creates parent chunks (4000 chars) and child chunks (200 chars)
3. Stores child chunks in ChromaDB for vector search
4. Stores parent chunks in InMemoryByteStore
5. On query, searches for the most relevant child chunks
6. Returns the corresponding parent chunks as context

### Reranker RAG (`reranker_rag.ipynb`)

**Characteristics:**
- **Simple Chunking**: Divides documents into 4000-character chunks with 20-character overlap
- **Two-Stage Retrieval**:
  1. Naive Retriever: Searches for the 10 most similar documents
  2. Cohere Rerank: Reorders and selects the top 3
- **Advantage**: Better precision through semantic reordering
- **Rerank Model**: Cohere rerank-v3.5

**Flow:**
1. Loads PDF and splits into 4000-character chunks
2. Stores chunks in ChromaDB
3. On query, retrieves 10 candidate documents
4. Uses Cohere Rerank to reorder and select the top 3
5. Passes compressed context to the LLM

## Technologies Used

- **LangChain**: Framework for developing applications with LLMs
- **OpenAI GPT-3.5-turbo**: Language model for response generation
- **OpenAI Embeddings**: Vector embedding generation
- **ChromaDB**: Vector database
- **Cohere Rerank**: Semantic reordering model
- **PyPDF**: PDF text extraction
- **Python 3.12**: Programming language

## Installation

### Prerequisites

```bash
sudo apt update
sudo apt install python3.12-venv
```

### Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install ipykernel
```

## License
This project is licensed under the terms specified in the LICENSE file.