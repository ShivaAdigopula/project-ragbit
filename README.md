# Enterprise RAG System

A production-grade, highly modular, and secure Retrieval-Augmented Generation (RAG) system built with FastAPI, MongoDB Vector Search, MarkItDown, Pydantic AI, Ollama (Gemma 4 & nomic-embed-text), and Streamlit.

---

## 🏗️ Phase 0: Project Initiation

This document establishes the architecture decisions, folder structure, coding guidelines, and development standards for the Enterprise RAG system.

---

## 🎯 Tech Stack Decisions & Rationale

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **API Layer** | **FastAPI** | High performance asynchronous execution, type safety with Pydantic, auto-generated OpenAPI documentation, and native dependency injection. |
| **Vector DB** | **MongoDB Vector Search** | Combines operational database capabilities with semantic vector search in a unified data store. Allows hybrid queries (metadata filters + vector search) without double-writing or syncing issues. |
| **Ingestion Engine** | **MarkItDown** | Microsoft's native text and formatting extractor. Converts PDFs, Office docs, etc., to structured Markdown, preserving semantic markers like headers and tables for superior chunking. |
| **Embeddings** | **LangChain OpenAIEmbeddings** | OpenAI-compatible hosted embeddings using NVIDIA's `nv-embedcode-7b-v1` (4096-dimensional vectors). |
| **Orchestration Agent** | **LangChain (ChatNVIDIA)** | Core agent and LLM orchestration layer. Utilizes `ChatNVIDIA` and `with_structured_output` to execute structured schema validation and RAG answer formatting. |
| **Reasoning Agent LLM** | **Nemotron 3 Ultra 550B** | LangChain-compatible ChatNVIDIA hosted model. Nemotron-3-Ultra-550B provides state-of-the-art logical reasoning and strict schema conformance. |
| **Frontend UI** | **Streamlit** | Enables swift development of internal dashboards, chat inputs, and source document displays using clean Python code. |


---

## 📁 Repository Directory Structure

The system follows a clean architecture pattern, decoupling the api delivery mechanism, business services, retrieval logic, and database access.

```text
Project-Ragbit/
├── backend/
│   ├── api/          # FastAPI routers, endpoints, custom exception handlers, and middlewares
│   ├── core/         # Config loader (Pydantic Settings), logging setups, security utilities
│   ├── db/           # MongoDB database client and Repository pattern classes
│   ├── ingestion/    # Document conversion (MarkItDown), preprocessing, and chunking strategy
│   ├── models/       # Pydantic schemas (request/response) and Database Models (ODM)
│   ├── rag/          # Query preprocessing, embedding models, vector search, and ranking logic
│   ├── services/     # Pydantic AI agent, prompt construction, LLM orchestration, and reasoning
│   ├── utils/        # Generic helpers, timestamp formatters, cleanups
│   └── main.py       # FastAPI application entrypoint
├── frontend/
│   ├── app.py        # Streamlit app entrypoint
│   └── components/   # Modular UI layout elements (sidebar, chat history, document viewer)
├── config/           # Static application configs (YAML/JSON)
├── tests/            # Test suite separated into unit, integration, and performance tests
├── .gitignore        # Version control exclude files
├── README.md         # Documentation base
└── requirements.txt  # Project dependencies list
```

---

## 📏 Coding Standards & Architecture Principles

To build a production-grade system, the following standards are strictly enforced:

### 1. SOLID Principles
* **Single Responsibility Principle (SRP):** Each class/module must have only one reason to change. E.g., the parser converts documents, the embedder creates vectors; they must not mix.
* **Open/Closed Principle (OCP):** System components (like chunkers or retrieval rankers) should be open for extension but closed for modification via abstract interfaces.
* **Liskov Substitution Principle (LSP):** Subtypes must be substitutable for their base types (e.g., swapping embedding providers).
* **Interface Segregation Principle (ISP):** Clients should not be forced to depend on methods they do not use.
* **Dependency Inversion Principle (DIP):** Depend on abstractions, not concretions. Inject dependencies (like DB clients or LLM services) using FastAPI's dependency injection.

### 2. Clean Architecture
* **Decoupled Layers:** The database layer (`backend/db`) does not know about FastAPI routers. The API layer (`backend/api`) only interacts with schemas and services.
* **No Framework Bloat:** Do NOT use high-abstraction libraries unless explicitly needed. Custom wrapper integrations with LangChain (`ChatNVIDIA` / `OpenAIEmbeddings`) and MongoDB keep the execution path explicit, debuggable, and performant.

### 3. Coding Guidelines
* **Type Hints:** All function definitions must have type signatures. Use Python 3.10+ type syntax (`dict[str, Any]`, `str | None`).
* **Pydantic Validation:** All incoming data payloads (API requests, config profiles, ingestion payloads) must be validated via Pydantic v2 schemas.
* **Asynchronous execution:** Use `async/await` for IO-bound operations (FastAPI endpoints, HTTP clients, MongoDB Atlas queries using Motor).
* **Structured Logging:** Log events in JSON format with contextual attributes (Correlation IDs, query duration, model confidence). Avoid stdout `print` statements.

---

## 🗺️ Project Roadmap (Phased Execution)

* [x] **Phase 0:** Project Initiation & Standards setup
* [x] **Phase 1:** Software Requirements Specification (SRS)
* [x] **Phase 2:** High-Level & Low-Level System Design
* [x] **Phase 3:** Directory Scaffolding & Logging / Config Setup
* [x] **Phase 4:** MarkItDown Document Ingestion Pipeline
* [x] **Phase 5:** MongoDB Vector DB Storage Integration
* [x] **Phase 6:** Semantic & Hybrid Retrieval Engine
* [x] **Phase 7:** LangChain ChatNVIDIA Orchestration Layer
* [x] **Phase 8:** LangChain Structured Output Generation
* [x] **Phase 9:** FastAPI Service Layer
* [x] **Phase 10:** Streamlit Frontend
* [x] **Phase 11:** Final Cleanup, Documentation, & Validation

---

## 🔧 Installation & Local Setup Guide

Follow these steps to configure and run the application in a local development environment.

### 1. System Prerequisites
* **Python:** Version 3.10 or higher.
* **MongoDB:** An active instance of MongoDB Community Server running locally or an Atlas connection string.
* **Ollama:** Installed locally (macOS/Linux/Windows). Download from [ollama.com](https://ollama.com).

### 2. Model Setup
Both the generative reasoning model (`nvidia/nemotron-3-ultra-550b-a55b`) and the embedding model (`nvidia/nv-embedcode-7b-v1`) are hosted externally and accessed via OpenAI-compatible endpoints using the NVIDIA API. No local LLM download or running Ollama is required.

### 3. Project Configuration & Installation
1. Clone the repository and navigate to the project root directory:
   ```bash
   cd Project-Ragbit
   ```
2. Initialize and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the application dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Create a local environment file `.env` in the project root:
   > [!IMPORTANT]
   > If your MongoDB password or username contains special characters (such as `@`, `:`, `/`, `+`, etc.), they **must be URL-encoded** (for example, `@` becomes `%40`) in the connection string to prevent URI parsing and authentication errors.

   ```env
    # Database Configurations
    MONGODB_URI=mongodb://localhost:27017
    MONGODB_DB_NAME=rag_enterprise_db

    # NVIDIA API Configurations
    NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
    NVIDIA_API_KEY=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
    NVIDIA_GEN_MODEL=nvidia/nemotron-3-ultra-550b-a55b
    NVIDIA_EMBED_API_KEY=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
    NVIDIA_EMBED_MODEL=nvidia/nv-embedcode-7b-v1
    EMBEDDING_PROVIDER=nvidia
    EMBEDDING_DIMENSION=4096

   # Logging Configurations
   LOG_LEVEL=INFO
   ```

### 4. Running the Application

#### A. Run the FastAPI Backend:
Ensure MongoDB is running, then start the FastAPI development server:

```bash
# Option 1: Run from the project root directory (Recommended)
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000

# Option 2: Alternatively, navigate to the backend directory
cd backend
uvicorn main:app --reload --port 8000
```
Verify the API liveness by visiting `http://localhost:8000/health` or the OpenAPI documentation at `http://localhost:8000/docs`.

#### B. Run the Streamlit Frontend:
In a separate terminal (with the virtual environment activated):
```bash
cd frontend
streamlit run app.py --server.port 8501
```
Open `http://localhost:8501` in your browser to interact with the RAG user interface.

