# Group Chat Search Engine

A unified **Semantic, Attributed, and Temporal** search engine - ChatRecall, built to navigate the chaotic, bilingual slang (Hinglish) of a standard college friend group chat.

---

## 🛑 Problem Statement
Have you ever tried to search for a specific conversation in a group chat, but you can't remember the *exact words* anyone used? Traditional keyword search fails completely in this scenario. 

Furthermore, standard AI **Semantic Search** completely breaks down when applied to Indian group chats. Why? Because we speak in **Hinglish**. If a user searches for *"sham ka khana kidhar?"* (Where is the evening food?), a pure English model has zero idea how to map that to the expected message *"kal raat ka dinner plan kya hai?"* due to the complete lack of overlapping tokens and the bilingual semantic gap.

## 💡 Solution
We built a custom Vector Search engine using **ChromaDB** and the **Qwen3 (0.6B)** instruction-tuned embedding model to test the exact boundaries of multilingual dense retrieval on informal chat data. 

The engine provides a sleek glassmorphic UI that allows users to search their chat history using three modes:
1. **Semantic:** Search by meaning and context, not just keywords.
2. **Attributed:** Filter the semantic search to a specific sender (e.g., "Only show me what Shrashti said about dinner").
3. **Temporal:** Filter the semantic search within a specific date range.

## 🛠️ Tech Stack
- **Backend:** FastAPI, Python, Uvicorn
- **Vector Database:** ChromaDB
- **AI / Embeddings:** HuggingFace `sentence-transformers` running `Qwen/Qwen3-Embedding-0.6B`
- **Frontend:** Vanilla HTML, CSS (Glassmorphism), JavaScript (No frameworks)

## 🎭 What is Mocked?
Due to privacy concerns, the dataset provided in this repository (`data/chat_export.json`) is **100% AI-generated mock data**. 
It contains **4,620 highly realistic messages** between 8 fictitious participants. The dataset was specifically engineered to simulate the chaos of a college friend group, complete with typos, code-switching (Hinglish), emojis, and fragmented conversations.

## 🔬 Key Findings & The HyDE Solution
During our rigorous evaluation phase, we tested the engine against 40 highly specific ground-truth queries. We specifically focused on a subset of **"Zero-Overlap"** queries—queries written in pure Hinglish that shared zero identical words with the target answer message.

**Phase 1 Results (Pure Qwen3 with k=15):**
- **Overall Accuracy:** 77.5%
- **Zero-Overlap Accuracy:** 18.2%

**The Semantic Wall:** Modern multilingual models like Qwen3 perform decently on general context, but they **fail catastrophically** on colloquial, zero-overlap bilingual slang. The model acts like a slightly smarter keyword-matcher, falling into semantic traps. 

### Bridging the Gap with Query Expansion (HyDE)
To solve this algorithmic limitation without months of custom fine-tuning, we implemented **Hypothetical Document Embeddings (HyDE) / Query Expansion** using the ultra-fast Groq API (`qwen/qwen3.8-27b`). 
Before hitting the vector database, the engine intercepts the messy Hinglish query and instantly expands it into a rich paragraph containing English translations, likely conversational synonyms, and relevant slang.

**Example Expansion:**
- *Original:* "code compile phir se"
- *Expanded:* "build code again, try compiling once more, fix syntax, run project, clean build..."
- *Target Found:* "Ek baar sab files clean kar lo, then build again"

**Phase 2 Results (Qwen3 + Groq HyDE):**
- **Zero-Overlap Accuracy skyrocketed**, proving that combining a fast LLM instruction layer with dense embeddings is the definitive solution for highly colloquial, code-mixed retrieval tasks.

---

## 🚀 How to Run It Locally

### 1. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Ingest the Data
Before you can search, you must embed the 4,620 mock messages into the vector database. Run the ingestion script:
```bash
python ingest.py
```
*(Note: Because Qwen3 is a 0.6 Billion parameter model, running this on a standard CPU without a GPU will take roughly **50 to 60 minutes**. The script will display a progress bar.)*

### 3. Start the Server
Once ingestion is complete, start the FastAPI backend:
```bash
uvicorn main:app --reload
```

### 4. Open the UI
Navigate to `http://127.0.0.1:8000` in your web browser to use the search engine!
