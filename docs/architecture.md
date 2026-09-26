# Architectural Specification — AI Knowledge Assistant (PostgreSQL + RAG + MCP)

## Overview

The **AI Knowledge Assistant** is a production-grade, hybrid database AI system incorporating:
- **PostgreSQL 18**: Relational persistence for expenses, document metadata, conversations, and messages using SQLAlchemy 2.x, Psycopg 3, and Alembic migrations.
- **ChromaDB**: Persistent vector database storing 384-dimensional dense embeddings (`all-MiniLM-L6-v2`) for semantic search and document RAG.
- **MCP Subsystem**: Model Context Protocol tool registry executing local tools (`add_expense`, `list_expenses`, `summarize_expenses`, `get_stock_price`, `duckduckgo_search`).
- **Structured Output Pipeline**: Pydantic validation & relative date resolution for natural language inputs.

---

## Architectural Diagram

```mermaid
graph TD
    User([User Request]) --> UI[Streamlit UI]
    UI --> Core[Assistant Core & Router]
    Core --> Intent{Intent Detection}
    
    Intent -->|General Chat| LLM[Groq LLM Llama 3.3 70B]
    Intent -->|Document RAG| RAG[Hybrid Retriever & Reranker]
    Intent -->|Expense / Tools| MCP[MCP Tool Registry]

    RAG --> Chroma[(ChromaDB Vector Store)]
    MCP --> Service[Expense Service]
    Service --> Repo[Expense Repository]
    Repo --> ORM[SQLAlchemy 2.x]
    ORM --> Postgres[(PostgreSQL 18 Database)]

    Postgres -->|Tables| T1[expenses]
    Postgres -->|Tables| T2[documents]
    Postgres -->|Tables| T3[conversations]
    Postgres -->|Tables| T4[messages]
```

---

## Database ERD (Entity Relationship Diagram)

```mermaid
erDiagram
    conversations ||--o{ messages : "contains"
    conversations {
        string id PK
        string title
        datetime created_at
        datetime updated_at
    }
    messages {
        string id PK
        string conversation_id FK
        string role
        text content
        datetime created_at
    }
    expenses {
        string id PK
        numeric amount
        string category
        string description
        date expense_date
        datetime created_at
        datetime updated_at
    }
    documents {
        string id PK
        string filename
        string file_hash
        string file_type
        bigint file_size
        integer page_count
        integer chunk_count
        string status
        datetime uploaded_at
    }
```
