# AI Knowledge Assistant — RAG + MCP + PostgreSQL

A production-grade AI assistant combining **PostgreSQL 18** relational database persistence, **ChromaDB** vector search, **MCP (Model Context Protocol)** tool execution, **Groq Llama 3.3 70B** LLM orchestration, and an interactive **Streamlit** user interface.

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User Prompt]) --> UI[Streamlit UI]
    UI --> Router[Query Router & Intent Classifier]
    
    Router -->|GENERAL| LLM[Groq LLM]
    Router -->|DOCUMENT| RAG[Hybrid RAG & Reranker]
    Router -->|TOOL / EXPENSE| MCP[MCP Expense Tools]

    RAG --> Chroma[(ChromaDB Vector Store)]
    MCP --> Service[Expense Service Layer]
    Service --> Repo[SQLAlchemy 2.x Repository]
    Repo --> Postgres[(PostgreSQL 18 Database)]
```

---

## ⚡ Key Features

- **PostgreSQL 18 Database Integration**: Uses SQLAlchemy 2.x and Psycopg 3 for relational persistence across `expenses`, `documents`, `conversations`, `messages`, and `tool_executions`.
- **Alembic Database Migrations**: Automated schema migration management via `alembic upgrade head`.
- **Pydantic Structured Output Parsing**: Validates expense requests (`amount > 0`, `category`, `description`, `expense_date`) and computes relative dates (`yesterday`, `today`, `Monday`) dynamically.
- **MCP Tool Protocol**: Self-contained `add_expense`, `list_expenses`, `summarize_expenses`, `update_expense`, `delete_expense` tools backed by PostgreSQL (with local JSON fallback), alongside DuckDuckGo Web Search & Yahoo Finance stock data tools.
- **Hybrid RAG Pipeline**: SentenceTransformers (`all-MiniLM-L6-v2`) embeddings in ChromaDB, query transformation, multi-signal reranking, evidence confidence scoring, and traceable source citations.
- **Prompt Security & Validation**: Built-in prompt guard sanitization and input validation against prompt injection.
- **Observability & Health Checks**: Reusable `database_health_check()` displaying live PostgreSQL connectivity (`PostgreSQL: ✓ Connected`) in the Streamlit UI telemetry drawer.

---

## 📁 Project Directory Structure

```
.
├── app.py                      # Streamlit application entrypoint
├── alembic.ini                 # Alembic configuration
├── data/                       # Sample data & local JSON storage fallback
├── docs/                       # Project documentation & reference specs
├── migrations/                 # Alembic migration scripts
├── scripts/                    # Maintenance & data migration utilities
├── tests/                      # Suite of 41 unit and integration tests
└── src/
    ├── assistant/              # Assistant state & orchestrator
    ├── chat/                   # Conversation history & chat handlers
    ├── database/               # PostgreSQL connection, models & repositories
    ├── evaluation/             # RAG evaluation metrics & benchmarks
    ├── llm/                    # Groq LLM integration & prompt templates
    ├── mcp/                    # MCP tools, registry, parser & store
    ├── memory/                 # Short & long-term conversation memory
    ├── rag/                    # Ingestion, retrieval, reranking & citations
    ├── schemas/                # Pydantic data models & contracts
    ├── security/               # Prompt guard & input sanitization
    ├── services/               # Expense, Document & Conversation domain services
    ├── ui/                     # Streamlit UI components & layout
    └── utils/                  # Helper utilities & logging
```

---

## 🛠️ PostgreSQL Setup & Migration Instructions

### STEP 1: Install PostgreSQL 18
Ensure PostgreSQL 18 is installed on your system and running on `localhost:5432`.

### STEP 2: Create Database
Open pgAdmin or psql and execute:
```sql
CREATE DATABASE ai_knowledge_assistant;
```

### STEP 3: Setup Virtual Environment
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### STEP 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### STEP 5: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Set your `GROQ_API_KEY` and `DATABASE_URL`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/ai_knowledge_assistant
```

### STEP 6: Run Database Migrations
Run Alembic schema migrations:
```bash
alembic upgrade head
```

### STEP 7: (Optional) Migrate Existing JSON Expense Data
If you have existing expenses in `data/expenses.json`, run the migration utility:
```bash
python scripts/migrate_expenses.py
```

### STEP 8: Launch Application
```bash
streamlit run app.py
```

---

## 🧪 Testing

Run all unit and integration tests (41 test cases passing across RAG, MCP tools, PostgreSQL models, hybrid routing, security prompt guard, and evaluation):

```bash
pytest
```

---

## 💻 Example Queries

- **Expense Addition**: `"add an expense of 200 for shopping yesterday for snake and chips"`
- **Expense Listing**: `"show my expenses"`
- **Expense Summarization**: `"how much did I spend on shopping this month?"`
- **Document Q&A (RAG)**: `"What does my uploaded PDF say about differential privacy?"`
- **Stock Lookup**: `"What is the current stock price of Apple?"`
- **Web Search**: `"Search the web for latest RAG benchmarks."`
- **Hybrid Query**: `"Compare my uploaded report with the latest news online."`
