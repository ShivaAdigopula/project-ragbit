# Graph Report - /Users/shivachandu/Personal Workspace/Projects/Project-Ragbit  (2026-07-06)

## Corpus Check
- Corpus is ~11,392 words - fits in a single context window. You may not need a graph.

## Summary
- 183 nodes · 314 edges · 19 communities
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Query Interface|Query Interface]]
- [[_COMMUNITY_Base Route Configuration|Base Route Configuration]]
- [[_COMMUNITY_Heading-Aware Document Chunker|Heading-Aware Document Chunker]]
- [[_COMMUNITY_Document CRUD & Cascade Deletes|Document CRUD & Cascade Deletes]]
- [[_COMMUNITY_Semantic Embeddings Service|Semantic Embeddings Service]]
- [[_COMMUNITY_LLM Agent & Orchestration|LLM Agent & Orchestration]]
- [[_COMMUNITY_MongoDB Database Client|MongoDB Database Client]]
- [[_COMMUNITY_Background Ingestion Processor|Background Ingestion Processor]]
- [[_COMMUNITY_Configuration Management|Configuration Management]]

## God Nodes (most connected - your core abstractions)
1. `RAGRepository` - 23 edges
2. `RetrievalEngine` - 18 edges
3. `PydanticAIEngine` - 14 edges
4. `DocumentChunker` - 12 edges
5. `ContextCompressor` - 12 edges
6. `DatabaseClient` - 10 edges
7. `IngestionPipeline` - 10 edges
8. `OllamaEmbedder` - 10 edges
9. `execute_query()` - 9 edges
10. `PromptBuilder` - 9 edges

## Surprising Connections (you probably didn't know these)
- `test_repository_get_document_by_hash()` --calls--> `RAGRepository`  [EXTRACTED]
  tests/test_db.py → backend/db/repository.py
- `test_repository_insert_document()` --calls--> `RAGRepository`  [EXTRACTED]
  tests/test_db.py → backend/db/repository.py
- `test_chunker_basic_splitting()` --calls--> `DocumentChunker`  [EXTRACTED]
  tests/test_ingestion.py → backend/ingestion/chunker.py
- `test_chunker_sliding_window()` --calls--> `DocumentChunker`  [EXTRACTED]
  tests/test_ingestion.py → backend/ingestion/chunker.py
- `test_keyword_score_calculation()` --calls--> `RetrievalEngine`  [EXTRACTED]
  tests/test_retrieval.py → backend/rag/search.py

## Import Cycles
- None detected.

## Communities (19 total, 0 thin omitted)

### Community 0 - "Query Interface"
Cohesion: 0.12
Nodes (23): Citation, execute_query(), QueryRequest, QueryResponse, QueryResponseMetadata, Executes a hybrid RAG query. Retrieves semantic chunks from MongoDB Atlas,     c, CitationSchema, PydanticAIEngine (+15 more)

### Community 1 - "Base Route Configuration"
Cohesion: 0.12
Nodes (19): liveness_probe(), Liveness probe to confirm FastAPI service is running., Sets up global application logging hierarchy., Custom formatter to output structured logs or clean console logs., setup_logging(), StructuredFormatter, DatabaseClient, Closes the active database connection pool. (+11 more)

### Community 2 - "Heading-Aware Document Chunker"
Cohesion: 0.09
Nodes (16): DocumentChunker, Any, Estimates token count based on word metrics., Splits text into sliding windows based on max token rules and overlap percentage, Splits Markdown text into blocks based on headings, then subdivides blocks, Splits Markdown text into semantic chunks based on headings and token limits., DocumentParser, Saves file bytes to a temporary directory in the workspace,          converts th (+8 more)

### Community 3 - "Document CRUD & Cascade Deletes"
Cohesion: 0.10
Nodes (17): delete_document(), get_document_status(), Retrieves the current metadata and processing status of a document., Deletes the document and cascades down to remove all of its associated vector ch, Any, RAGRepository, Retrieves raw chunks filtering by source filename., Executes a vector search query against the MongoDB document_chunks collection. (+9 more)

### Community 4 - "Semantic Embeddings Service"
Cohesion: 0.13
Nodes (13): OllamaEmbedder, Generates a 768-dimensional float embedding vector for a given query or chunk of, Generates embeddings for a batch of text chunks sequentially (or concurrently)., Client for generating semantic vectors locally via Ollama's nomic-embed-text mod, Any, Search and retrieval coordinator. Integrates vector search queries      with a c, Cleans and splits text into tokens, removing standard stop words., Calculates the keyword density match score between the query tokens and chunk te (+5 more)

### Community 5 - "LLM Agent & Orchestration"
Cohesion: 0.18
Nodes (7): Any, Executes structured generation using local LLM orchestrated by Pydantic AI., Any, Selects top-scoring chunks that fit within the context token budget., System prompt enforcing strict grounding and citation rules., Constructs the user payload containing context blocks and the user query., test_prompt_builder_structure()

### Community 6 - "MongoDB Database Client"
Cohesion: 0.22
Nodes (6): AsyncIOMotorClient, Readiness probe to confirm all backend services (MongoDB, Ollama) are connected., readiness_probe(), Returns the active database client, initializing if necessary., Helper to get database instance directly., Response

### Community 7 - "Background Ingestion Processor"
Cohesion: 0.33
Nodes (6): process_and_embed_document_task(), Asynchronous background task to execute parsing, chunking,     embedding generat, Uploads a document to start the ingestion pipeline.     If the document has been, upload_document(), BackgroundTasks, UploadFile

### Community 8 - "Configuration Management"
Cohesion: 0.67
Nodes (3): Application Settings validated using Pydantic Settings.     Reads environment va, Settings, BaseSettings

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `RAGRepository` connect `Document CRUD & Cascade Deletes` to `Base Route Configuration`, `Semantic Embeddings Service`, `MongoDB Database Client`, `Background Ingestion Processor`?**
  _High betweenness centrality (0.268) - this node is a cross-community bridge._
- **Why does `RetrievalEngine` connect `Semantic Embeddings Service` to `Query Interface`, `Document CRUD & Cascade Deletes`?**
  _High betweenness centrality (0.182) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `RAGRepository` (e.g. with `DatabaseClient` and `RetrievalEngine`) actually correct?**
  _`RAGRepository` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `RetrievalEngine` (e.g. with `Citation` and `QueryRequest`) actually correct?**
  _`RetrievalEngine` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `PydanticAIEngine` (e.g. with `Citation` and `QueryRequest`) actually correct?**
  _`PydanticAIEngine` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `ContextCompressor` (e.g. with `Citation` and `QueryRequest`) actually correct?**
  _`ContextCompressor` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Asynchronous background task to execute parsing, chunking,     embedding generat`, `Uploads a document to start the ingestion pipeline.     If the document has been`, `Retrieves the current metadata and processing status of a document.` to the rest of the system?**
  _50 weakly-connected nodes found - possible documentation gaps or missing edges._