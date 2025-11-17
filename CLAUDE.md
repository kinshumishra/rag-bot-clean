# CLAUDE.md - AI Assistant Reference Guide

## Project Overview

**Project Name:** RAG Bot Clean
**Author:** Siddharth Mishra
**Purpose:** A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content using vector embeddings and LLM-powered answers.

**Current Branch:** `claude/claude-md-mi2nydpaqv0f26he-01Gfx8h9HednAiZ38J9kirNL`

---

## Technology Stack

### Core Technologies
- **Language:** Python 3.11
- **Web Framework:** Streamlit (interactive web UI)
- **LLM Orchestration:** LangChain ecosystem
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **PDF Processing:** pypdf
- **Embeddings:** HuggingFace Sentence Transformers
- **LLM Provider:** Groq API

### Key Dependencies
```
streamlit              # Web framework for interactive UI
langchain              # LLM orchestration framework
langchain-core         # Core LangChain abstractions
langchain-community    # Community integrations
langchain-groq         # Groq API integration
pypdf                  # PDF parsing and loading
sentence-transformers  # HuggingFace embeddings
faiss-cpu              # Vector similarity search
groq                   # Groq API client
python-dotenv          # Environment variable management
```

---

## Directory Structure

```
/home/user/rag-bot-clean/
├── ui_app.py                  # ✅ PRODUCTION - Main Streamlit application
├── rag-bot.py                 # 🔧 LEGACY - Original implementation (deprecated imports)
├── requirements.txt           # Python dependencies
├── Diabetes.pdf               # Sample test data (1.5MB)
├── temp.pdf                   # Temporary upload storage (generated at runtime)
├── .devcontainer/
│   └── devcontainer.json     # Dev container configuration
├── .git/                      # Git version control
├── .gitignore                 # Ignores: venv/, __pycache__/, *.pyc
├── a/                         # Virtual environment (venv)
└── create/                    # Virtual environment (venv)
```

---

## Key Files

### Primary Application Files

#### `ui_app.py` (PRODUCTION - USE THIS)
- **Location:** `/home/user/rag-bot-clean/ui_app.py`
- **Status:** Current production version
- **Purpose:** Main Streamlit application with RAG functionality
- **Lines:** 73 lines
- **Key Features:**
  - Uses modern LangChain imports (`langchain_text_splitters`, `langchain_core.messages`)
  - Groq LLM integration with `openai/gpt-oss-20b` model
  - FAISS vector store with HuggingFace embeddings
  - Streamlit form-based UI with file upload
  - Shows top 3 matching document chunks

**Important Code Sections:**
- Lines 11-14: Embedding model initialization
- Lines 15-22: Groq LLM setup with API key handling
- Lines 34-43: PDF processing and vectorization pipeline
- Lines 54-64: Query handling and LLM response generation

#### `rag-bot.py` (LEGACY - DEPRECATED)
- **Location:** `/home/user/rag-bot-clean/rag-bot.py`
- **Status:** Old version with deprecated imports
- **Purpose:** Original implementation, kept for reference
- **Warning:** Uses `langchain.text_splitter` and `langchain.schema` (deprecated)

### Configuration Files

#### `requirements.txt`
- Lists 12 Python packages
- No version pinning (could cause dependency conflicts)

#### `.devcontainer/devcontainer.json`
- Python 3.11 Bookworm base image
- Auto-installs requirements on container creation
- Launches Streamlit on port 8501 with CORS disabled
- Opens README.md and ui_app.py on startup

---

## Architecture & RAG Pipeline

### Complete RAG Workflow

```
┌─────────────────┐
│  User Uploads   │
│    PDF File     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  PyPDFLoader                    │
│  Extracts text from PDF         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  RecursiveCharacterTextSplitter │
│  chunk_size=500                 │
│  chunk_overlap=100              │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  HuggingFaceEmbeddings          │
│  all-MiniLM-L6-v2              │
│  Converts chunks to vectors     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  FAISS VectorStore              │
│  Indexes embeddings             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  User Submits   │
│     Query       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Similarity Search (k=3)        │
│  Retrieves top 3 chunks         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Context Assembly               │
│  Joins chunks with \n\n         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  ChatGroq (Groq LLM)           │
│  Model: openai/gpt-oss-20b     │
│  Temperature: 0.7               │
│  Max Tokens: 2000               │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Display Answer + Source Chunks │
└─────────────────────────────────┘
```

### Key Parameters

| Component | Parameter | Value | Purpose |
|-----------|-----------|-------|---------|
| Text Splitter | `chunk_size` | 500 | Characters per chunk |
| Text Splitter | `chunk_overlap` | 100 | Overlap between chunks |
| Embeddings | `model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Fast, efficient embeddings |
| LLM | `model` | `openai/gpt-oss-20b` | Groq-hosted open model |
| LLM | `temperature` | 0.7 | Balance creativity/consistency |
| LLM | `max_tokens` | 2000 | Maximum response length |
| Vector Search | `k` | 3 | Number of chunks to retrieve |

---

## Development Setup

### Local Development

1. **Prerequisites:**
   - Python 3.11
   - Groq API key (required)

2. **Installation:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   - `GROQ_API_KEY` - Required for LLM functionality
   - Can be set via:
     - Streamlit secrets: `.streamlit/secrets.toml`
     - Environment variable: `export GROQ_API_KEY=your_key`

4. **Running the Application:**
   ```bash
   streamlit run ui_app.py
   ```

### Dev Container Setup

The project includes a `.devcontainer` configuration for VS Code/GitHub Codespaces:

- **Base Image:** `mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`
- **Auto-install:** Runs `pip3 install -r requirements.txt` on creation
- **Auto-start:** Launches Streamlit on port 8501
- **Extensions:** Python, Pylance
- **Port Forwarding:** 8501 (auto-opens preview)

**Dev Container Features:**
- CORS disabled for local development
- XSRF protection disabled for easier testing
- Auto-opens README.md and ui_app.py

---

## Code Conventions & Patterns

### Import Organization

**ALWAYS use these modern imports (from ui_app.py):**
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
```

**AVOID deprecated imports (from rag-bot.py):**
```python
# ❌ DON'T USE
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage
```

### API Key Management

Current implementation (ui_app.py:15):
```python
groq_api_key = st.secrets["GROQ_API_KEY"] or os.getenv("GROQ_API_KEY")
```

**Pattern:** Try Streamlit secrets first, fall back to environment variable.

### File Upload Pattern

Temporary file pattern (ui_app.py:34-35):
```python
with open("temp.pdf", "wb") as f:
    f.write(pdf_file.read())
```

**Note:** Uses `temp.pdf` as temporary storage. No cleanup after processing.

### Message Construction

System message pattern (ui_app.py:58-62):
```python
messages = [
    SystemMessage(
        content="You are a helpful assistant answering only from the provided context."
    ),
    HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}"),
]
```

**Pattern:** System message sets constraints, Human message includes context + query.

### UI Conventions

- **Emojis:** Used throughout UI (🧠, 📄, 🔎, 💬, etc.)
- **Forms:** Uses `st.form()` to prevent multiple reloads
- **Spinners:** `st.spinner("Thinking... 🤔")` for loading states
- **Expanders:** `st.expander()` for showing source chunks
- **Success:** `st.success()` for displaying answers

---

## Important Notes for AI Assistants

### File Preferences
1. **Always edit `ui_app.py`** - This is the production version
2. **Ignore `rag-bot.py`** unless asked to migrate or reference it
3. **Never modify virtual environments** (`a/` and `create/` directories)

### API Keys
- Code expects `GROQ_API_KEY` in either Streamlit secrets or environment
- No validation or error handling for missing API keys
- API key is passed directly to `ChatGroq` constructor

### Common Issues to Watch For

1. **Import Deprecations:**
   - LangChain has updated import paths
   - Always use `langchain_text_splitters` not `langchain.text_splitter`
   - Always use `langchain_core.messages` not `langchain.schema`

2. **File Handling:**
   - `temp.pdf` is overwritten on each upload
   - No file cleanup implemented
   - No validation of PDF file integrity

3. **Vector Store:**
   - Rebuilt on every PDF upload (no persistence)
   - No caching mechanism
   - Could be slow for large documents

4. **Error Handling:**
   - Limited exception handling
   - No graceful degradation for API failures
   - No user feedback for errors

### Security Considerations

1. **API Key Exposure:**
   - Line 23 prints LLM class name (safe)
   - API key handled properly (not printed)

2. **File Upload:**
   - No file size validation
   - No malicious PDF detection
   - Temporary files not cleaned up

3. **CORS/XSRF:**
   - Both disabled in dev container (development only)
   - Should be enabled in production

---

## Common Tasks & Workflows

### Adding New Features

1. **Adding New LLM Provider:**
   - Import new LLM class from LangChain
   - Replace `ChatGroq` initialization (lines 17-22)
   - Update API key handling
   - Test with sample query

2. **Changing Embedding Model:**
   - Update `model_name` in `HuggingFaceEmbeddings` (line 13)
   - Consider model size vs. accuracy tradeoff
   - Test similarity search quality

3. **Modifying Chunk Parameters:**
   - Adjust `chunk_size` and `chunk_overlap` (line 40)
   - Smaller chunks = more precise, less context
   - Larger chunks = more context, less precise

4. **Changing Retrieved Context:**
   - Modify `k` parameter in `similarity_search()` (line 54)
   - More chunks = more context, slower processing
   - Fewer chunks = faster, less comprehensive

### Testing Workflow

**No automated tests exist.** Manual testing process:

1. Start Streamlit: `streamlit run ui_app.py`
2. Upload test PDF (e.g., `Diabetes.pdf`)
3. Submit test queries
4. Verify:
   - Answer relevance
   - Source chunks displayed
   - No errors in console

### Git Workflow

**Current branch:** `claude/claude-md-mi2nydpaqv0f26he-01Gfx8h9HednAiZ38J9kirNL`

**When making commits:**
```bash
git add .
git commit -m "Descriptive message"
git push -u origin claude/claude-md-mi2nydpaqv0f26he-01Gfx8h9HednAiZ38J9kirNL
```

**Recent commit patterns:**
- Recent commits have unclear messages ("fihdabda", "shfdu", etc.)
- Consider using descriptive commit messages
- Last significant commit: "Changing import" (0fb221f)

---

## Enhancement Opportunities

### High Priority
1. **Error Handling:** Add try/except blocks for API calls and file operations
2. **API Key Validation:** Check for missing keys before initializing LLM
3. **File Cleanup:** Delete `temp.pdf` after processing
4. **Requirements Pinning:** Add version numbers to `requirements.txt`

### Medium Priority
5. **Vector Store Persistence:** Cache embeddings to avoid reprocessing
6. **File Size Limits:** Validate PDF size before processing
7. **Progress Indicators:** Show progress during embedding generation
8. **Session State:** Preserve vector store across queries

### Low Priority
9. **Multiple File Support:** Allow multiple PDF uploads
10. **Export Functionality:** Allow downloading chat history
11. **Custom System Prompts:** Let users customize LLM instructions
12. **Model Selection:** UI to choose different LLM models

---

## Testing & Quality Assurance

### Current State
- **Unit Tests:** None
- **Integration Tests:** None
- **CI/CD:** None
- **Linting:** Not configured
- **Type Checking:** Not configured

### Recommended Testing Strategy

1. **Unit Tests:**
   - Test text splitting logic
   - Test embedding generation
   - Mock LLM responses

2. **Integration Tests:**
   - End-to-end PDF upload → query → response
   - Test with various PDF formats
   - Test error scenarios (missing API key, corrupted PDF)

3. **Manual Testing Checklist:**
   - [ ] Upload PDF successfully
   - [ ] View extracted text
   - [ ] Submit query and get answer
   - [ ] View source chunks
   - [ ] Test with different PDF sizes
   - [ ] Test with non-PDF files (should fail gracefully)
   - [ ] Test without API key (should show error)

---

## Deployment Considerations

### Environment Variables Required
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Streamlit Cloud Deployment
1. Add `GROQ_API_KEY` to Streamlit secrets
2. Ensure `requirements.txt` is complete
3. Set main file to `ui_app.py`
4. Enable public URL or authentication

### Resource Requirements
- **Memory:** ~1-2GB (for embedding model + FAISS)
- **Storage:** Minimal (temp files only)
- **CPU:** Moderate (embedding generation is CPU-intensive)
- **Network:** API calls to Groq (requires internet)

### Production Recommendations
1. Enable CORS and XSRF protection
2. Add rate limiting for API calls
3. Implement caching for embeddings
4. Add logging and monitoring
5. Set file size limits
6. Add user authentication if needed

---

## Debugging Guide

### Common Errors

**Error: "GROQ_API_KEY not found"**
- **Cause:** Missing API key in environment or secrets
- **Solution:** Set `GROQ_API_KEY` in `.streamlit/secrets.toml` or environment

**Error: "No module named 'langchain_text_splitters'"**
- **Cause:** Outdated LangChain installation
- **Solution:** Run `pip install --upgrade langchain langchain-community`

**Error: "Cannot load PDF"**
- **Cause:** Corrupted PDF or unsupported format
- **Solution:** Validate PDF file, check file permissions

**Error: "FAISS index error"**
- **Cause:** Empty document or no text extracted
- **Solution:** Ensure PDF contains text (not just images)

### Debugging Steps

1. **Check Streamlit logs:** Look for stack traces in console
2. **Verify API key:** Print (securely) whether key is loaded
3. **Test embeddings:** Verify embedding model loads correctly
4. **Test PDF loading:** Check if `temp.pdf` is created and readable
5. **Test vector store:** Verify FAISS index is built successfully

---

## Version History & Migration Notes

### Recent Changes (from git log)

**Latest:** `0fb221f - Changing import`
- Updated import statements to modern LangChain paths
- Migrated from deprecated imports in `rag-bot.py`
- Created production-ready `ui_app.py`

**Previous:** Groq integration (PR #1)
- Added Groq as LLM provider
- Replaced OpenAI with Groq in main flow

### Migration from rag-bot.py to ui_app.py

**Key Changes:**
1. `langchain.text_splitter` → `langchain_text_splitters`
2. `langchain.schema` → `langchain_core.messages`
3. Model changed from `llama3-8b-8192` to `openai/gpt-oss-20b`
4. Improved API key handling with fallback

**Why ui_app.py is preferred:**
- Uses current LangChain APIs
- Better structured code
- More robust error handling potential
- Active development focus

---

## Quick Reference

### Start Application
```bash
streamlit run ui_app.py
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set API Key (Linux/Mac)
```bash
export GROQ_API_KEY=your_key_here
```

### Set API Key (Streamlit)
Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_key_here"
```

### Key File Locations
- Main App: `ui_app.py`
- Dependencies: `requirements.txt`
- Dev Container: `.devcontainer/devcontainer.json`
- Temp Files: `temp.pdf` (auto-generated)
- Sample Data: `Diabetes.pdf`

---

## AI Assistant Guidelines

### When Working on This Codebase

1. **Always read `ui_app.py` first** before making changes
2. **Use modern LangChain imports** from `langchain_core` and `langchain_text_splitters`
3. **Test locally** with `streamlit run ui_app.py` before committing
4. **Check API key availability** before running
5. **Preserve existing UI/UX patterns** (emojis, forms, expanders)
6. **Add error handling** when modifying existing functionality
7. **Update this CLAUDE.md** if you make architectural changes

### What to Avoid

1. **Don't modify `rag-bot.py`** - it's deprecated
2. **Don't use deprecated imports** from old LangChain versions
3. **Don't remove existing functionality** without user confirmation
4. **Don't commit without testing** the Streamlit app locally
5. **Don't expose API keys** in code or logs
6. **Don't ignore existing code style** (emojis, formatting)

### When to Ask User

- Before changing LLM provider or model
- Before modifying chunk size/overlap significantly
- Before adding new dependencies
- Before changing UI layout substantially
- Before implementing caching or persistence
- Before adding authentication or security features

---

## Contact & Attribution

**Created by:** Siddharth Mishra
**Project Type:** RAG Chatbot POC/Demo
**License:** Not specified
**Last Updated:** 2025-11-17

---

**Document Version:** 1.0
**Generated:** 2025-11-17
**For:** AI Assistant Reference
