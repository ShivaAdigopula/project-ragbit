# Enterprise RAG System

A production-grade, highly modular, and secure Retrieval-Augmented Generation (RAG) system built with FastAPI, MongoDB Vector Search, MarkItDown, Pydantic AI, Ollama (Gemma 12B & nomic-embed-text), and Streamlit.

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
| **Embeddings** | **nomic-embed-text (Ollama)** | A local, high-performance 768-dimensional text embedding model with an 8k context window, ensuring local data privacy and excellent retrieval relevance. |
| **Orchestration Agent** | **Pydantic AI** | Enterprise-grade agent framework for building structured LLM interfaces. Guarantees schema validation, handles retry logic, and outputs strict JSON responses with confidence scores. |
| **Local LLM** | **Ollama (Gemma 12B)** | Runs locally to satisfy strict data privacy and compliance. Gemma 12B provides advanced reasoning, high instruction-following performance, and a large context length. |
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
* **No Framework Bloat:** Do NOT use high-abstraction frameworks (like LlamaIndex or LangChain). Custom integrations with Pydantic AI, MongoDB, and Ollama keep the execution path explicit, debuggable, and performant.

### 3. Coding Guidelines
* **Type Hints:** All function definitions must have type signatures. Use Python 3.10+ type syntax (`dict[str, Any]`, `str | None`).
* **Pydantic Validation:** All incoming data payloads (API requests, config profiles, ingestion payloads) must be validated via Pydantic v2 schemas.
* **Asynchronous execution:** Use `async/await` for IO-bound operations (FastAPI endpoints, HTTP clients, MongoDB Atlas queries using Motor).
* **Structured Logging:** Log events in JSON format with contextual attributes (Correlation IDs, query duration, model confidence). Avoid stdout `print` statements.

---

## 🗺️ Project Roadmap (Phased Execution)

* [x] **Phase 0:** Project Initiation & Standards setup
* [ ] **Phase 1:** Software Requirements Specification (SRS)
* [ ] **Phase 2:** High-Level & Low-Level System Design
* [ ] **Phase 3:** Directory Scaffolding & Logging / Config Setup
* [ ] **Phase 4:** MarkItDown Document Ingestion Pipeline
* [ ] **Phase 5:** MongoDB Vector DB Storage Integration
* [ ] **Phase 6:** Semantic & Hybrid Retrieval Engine
* [ ] **Phase 7:** Ollama Gemma 12B Orchestration Layer
* [ ] **Phase 8:** Pydantic AI Structured Output Generation
* [ ] **Phase 9:** FastAPI Service Layer
* [ ] **Phase 10:** Streamlit Frontend
* [ ] **Phase 11:** Final Cleanup, Documentation, & Validation
