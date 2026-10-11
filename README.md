# AI Knowledge Assistant — RAG + MCP + PostgreSQL

A production-grade AI assistant combining **PostgreSQL** relational database persistence, **ChromaDB** vector search, **MCP (Model Context Protocol)** tool execution, **Groq / OpenAI** LLM orchestration, and an interactive **Streamlit** user interface.

---

## 🏗️ System Architecture

```mermaid
graph TD
    User([User Prompt]) --> UI[Streamlit UI]
    UI --> Router[Query Router & Intent Classifier]
    
    Router -->|GENERAL| LLM[Groq / OpenAI LLM]
    Router -->|DOCUMENT| RAG[Hybrid RAG & Reranker]
    Router -->|TOOL / MCP| MCP[MCP Tools Registry]

    RAG --> Chroma[(ChromaDB Vector Store)]
    MCP --> Service[Domain Service Layer]
    Service --> Repo[SQLAlchemy 2.x Repositories]
    Repo --> Postgres[(PostgreSQL Database)]
```

---

## ⚡ Key Features

- **Modern Chat Input with Native `+` File Attachment**:
  - Clean Gemini-style input bar with an inline **`+` attach icon** directly inside the input prompt.
  - Multi-file uploading (`PDF`, `TXT`, `MD`, `DOCX`) with automatic text extraction, chunking, and ChromaDB vector indexing.
  - Clutter-free UI: zero awkward buttons or captions when no files are attached; attached files display as compact chips right above the prompt with single-click `🗑️ Detach All`.

- **Strict Conversation-Scoped Document RAG**:
  - **Zero Knowledge Bleed**: Document queries retrieve context *exclusively* from documents attached to the active conversation, completely preventing unrelated global documents from leaking into the context.
  - **Global Knowledge Base Linking**: Easily attach any previously indexed document from the database into the current conversation via the sidebar's **📂 Global Knowledge Base** menu.
  - **Safe Detachment**: Detaching files removes the conversation association in PostgreSQL without deleting global vector database embeddings.

- **Resilient Tool Execution & Error Recovery**:
  - **Open-Weights Model Compatibility**: Built-in automatic recovery (`_recover_failed_generation`) for models like `openai/gpt-oss-120b` that emit reasoning tokens (`<|channel|>commentary`), preventing `400 tool_use_failed` errors.
  - **Tool Deduplication**: Automatically deduplicates identical tool calls to prevent repeated executions and duplicate output cards.
  - **Collapsible Tool Details**: Technical outputs (tool name, latency, raw JSON result) are housed inside an expandable **🛠️ View Tool Details** container that stays neatly collapsed by default.

- **Multi-Conversation History**:
  - Full chat persistence in PostgreSQL (`conversations`, `messages`, `conversation_files`).
  - Real-time search filter across past conversations by title or keyword.
  - Instant conversation switching and deletion with cascading cleanup.
  - Smart chat guard: **➕ New Chat** is dynamically disabled when the current conversation is already empty to prevent redundant blank chats.

- **PostgreSQL Database Integration**:
  - Powered by SQLAlchemy 2.x and Psycopg 3 with decoupled, thread-safe session management (`expire_on_commit=False`).
  - Manages `expenses`, `documents`, `conversations`, `messages`, and `conversation_files`.
  - Schema versioning and automated migrations managed via **Alembic**.

- **MCP Tool Protocol & Financial Services**:
  - Integrated Model Context Protocol (MCP) registry supporting local and remote tools.
  - Built-in tools: `add_expense`, `list_expenses`, `summarize_expenses`, `update_expense`, `delete_expense`.
  - Real-time stock market data via Alpha Vantage (`get_stock_price`) and live web search via DuckDuckGo.

- **Hybrid RAG Pipeline**:
  - Dual semantic + keyword retrieval powered by SentenceTransformers (`all-MiniLM-L6-v2`) and ChromaDB.
  - Contextual reranking, query transformation, evidence confidence scoring (🟢 High, 🟡 Moderate, 🔴 Low), and traceable citations.

- **Application Access Verification & Session Gate**:
  - **Passcode Protection on Load**: Blocks unauthenticated access upon app startup with a sleek centered access verification screen, preventing unauthenticated access to database records, conversations, and document indexes.
  - **Environment-Controlled Access**: Configured via `APP_ACCESS_CODE` in `.env` (with Streamlit secrets fallback).
  - **Session Locking**: Seamlessly unlocks into `st.session_state` with a dedicated **🔒 Lock Application** button in the sidebar to re-lock at any time.

- **Prompt Security & Observability**:
  - Built-in Prompt Guard sanitizing user inputs against injection attacks.
  - System status drawer featuring live PostgreSQL health check (`database_health_check()`) and MCP tool registry discovery.
  - RAG Evaluation Suite computing Precision@K, Recall@K, MRR, and Faithfulness benchmarks.

---

## 📁 Project Directory Structure

```
.
├── app.py                      # Streamlit application entrypoint
├── alembic.ini                 # Alembic configuration
├── data/                       # Sample data & local storage fallback
├── docs/                       # Project documentation & reference specs
├── migrations/                 # Alembic migration scripts
├── scripts/                    # Maintenance & data migration utilities
├── tests/                      # Suite of unit and integration tests
└── src/
    ├── assistant/              # Knowledge Assistant orchestrator, state & recovery
    ├── chat/                   # Conversation history & chat handlers
    ├── database/               # PostgreSQL connection, models & repositories
    │   ├── connection.py       # Engine, sessionmaker & health checks
    │   ├── models.py           # SQLAlchemy ORM models
    │   └── repositories/       # Conversation, Expense, and Document repositories
    ├── evaluation/             # RAG evaluation metrics & benchmarks
    ├── llm/                    # LLM factory & prompt templates
    ├── mcp/                    # MCP tools, registry, parser & executor
    ├── memory/                 # Short & long-term conversation memory
    ├── rag/                    # Ingestion, retrieval, reranking & citations
    ├── schemas/                # Pydantic data models & contracts
    ├── security/               # Prompt guard & input sanitization
    ├── services/               # Expense, Document & Conversation domain services
    ├── ui/                     # Streamlit UI components & layout
    └── utils/                  # Helper utilities & logging
```

---

## 🛠️ PostgreSQL Setup & Quick Start

### STEP 1: Install & Start PostgreSQL
Ensure PostgreSQL is installed and running on your system (e.g., `localhost:5432`).

### STEP 2: Create Database
Open `psql` or pgAdmin and run:
```sql
CREATE DATABASE "rag-with-mcp";
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
Create or edit your `.env` file with your credentials:

# LLM Provider Keys
GROQ_API_KEY="gsk_your_groq_api_key_here"
GROQ_MODEL="llama-3.3-70b-versatile"
OPENAI_API_KEY="sk-proj-your_openai_api_key_here"

# Database Connection (Psycopg 3 Driver)
DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/rag-with-mcp

# Optional Tool APIs
ALPHA_VANTAGE_API_KEY="your_alphavantage_key"
TAVILY_API_KEY="your_tavily_key"
```

### STEP 6: Run Database Migrations
Apply Alembic database migrations:
```bash
alembic upgrade head
```

### STEP 7: Launch Application
```bash
streamlit run app.py
```
> When prompted on the landing screen, enter your configured `APP_ACCESS_CODE` (e.g. `1234`) and click **Unlock Assistant 🚀** to access the assistant.

---

## 🧪 Testing

Run the full test suite covering access gate protection, multi-conversation persistence, hybrid routing, MCP tool executor, RAG retrieval, and security prompt guards:

```bash
pytest
```

To run specific test suites:
```bash
pytest tests/test_access_gate.py tests/test_multi_conversation.py tests/test_hybrid_router.py tests/test_retrieval.py -v
```

---

## 💻 Example Queries

- **Stock Lookup (MCP Tool)**: `"What is the current stock price of Apple (AAPL)?"`
- **Commodity / Live Search**: `"What is today stockprice of gold?"`
- **Expense Addition**: `"Add an expense of ₹250 for lunch today"`
- **Expense Listing & Query**: `"Show all my expenses from this month"`
- **Document Q&A (RAG)**: `"What are the key findings mentioned in the attached PDF?"`
- **Web Search**: `"Search the web for recent advancements in RAG architecture"`
- **Hybrid Search**: `"Compare the numbers from my uploaded financial report with current market data"`

