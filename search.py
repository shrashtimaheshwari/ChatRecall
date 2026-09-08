import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import time
import config
from groq import Groq

load_dotenv()

class SearchEngine:
    def __init__(self):
        print(f"Initializing SearchEngine with model: {config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_collection(name=config.COLLECTION_NAME)
        print(f"Connected to ChromaDB collection: {config.COLLECTION_NAME}")
        
        # Determine the fixed "today" anchor date (max timestamp in the dataset)
        self.messages = []
        self.msg_id_to_idx = {}
        self.today_anchor = self._load_data_and_get_max_timestamp()
        print(f"Fixed 'today' anchor date set to: {self.today_anchor}")
        
        # Initialize Groq for query expansion
        if getattr(config, "ENABLE_QUERY_EXPANSION", False):
            print("Query Expansion is ENABLED. Initializing Groq client...")
            self.groq_client = Groq()
        else:
            self.groq_client = None

    def _load_data_and_get_max_timestamp(self) -> str:
        data_file = "data/chat_export.json"
        if not os.path.exists(data_file):
            return "2026-12-31T23:59:59Z"
            
        with open(data_file, "r", encoding="utf-8") as f:
            self.messages = json.load(f)
            
        self.msg_id_to_idx = {m["id"]: i for i, m in enumerate(self.messages)}
            
        if not self.messages:
            return "2026-12-31T23:59:59Z"
            
        # Assuming messages are sorted chronologically, the last one has the max timestamp
        return self.messages[-1]["timestamp"]

    def expand_query(self, query: str) -> str:
        if not self.groq_client:
            return query
            
        system_prompt = (
            "You are an expert search query expansion assistant for Indian college students. "
            "Your goal is to rewrite the user's messy Hinglish or English chat search query "
            "into a highly descriptive paragraph of likely keywords. You MUST include:\n"
            "1. The exact English translation.\n"
            "2. Very broad synonyms (e.g., if 'compile' -> add 'build, run, clean, error'. If 'padhai' -> add 'notes, study, exam, revision').\n"
            "3. Colloquial conversational words that might appear in the actual WhatsApp reply.\n"
            "Do NOT answer the query. Do NOT add conversational filler. ONLY return the expanded text paragraph.\n"
            "Example Query: 'sham ka khana kidhar'\n"
            "Example Output: 'Where is the evening food? What is the dinner plan? kal raat ka dinner plan kya hai? dinner food eat restaurant delivery'"
        )
        
        try:
            completion = self.groq_client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=200,
            )
            expanded = completion.choices[0].message.content.strip()
            if not expanded:
                expanded = query
            print(f"[Query Expansion]\n  Original: {query}\n  Expanded: {expanded}")
            return expanded
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return query

    def embed_query(self, query: str) -> tuple:
        # Step 1: Expand query if enabled
        expanded_query = self.expand_query(query)
        
        # Step 2: Add model-specific prefixes
        if "Qwen3" in config.EMBEDDING_MODEL:
            prefix = "Instruct: Given a search query, retrieve relevant chat messages that answer the query\nQuery: "
            emb = self.model.encode(prefix + expanded_query, show_progress_bar=False).tolist()
        else:
            emb = self.model.encode(expanded_query, show_progress_bar=False).tolist()
            
        return emb, expanded_query

    def search(
        self, 
        query: str, 
        mode: str = "semantic", 
        sender: Optional[str] = None, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        top_k: int = 5,
        context_n: int = 3
    ) -> Dict[str, Any]:
        
        start_time = time.time()
        query_embedding, expanded_query = self.embed_query(query)
        where_clause = None

        if mode == "attributed" and sender:
            where_clause = {"sender": sender}
            
        # Semantic mode requires no where_clause
        
        # If temporal, we must fetch more results and filter in Python 
        # because ChromaDB $gte/$lte operators only support ints/floats, not strings.
        fetch_k = top_k
        if mode == "temporal":
            fetch_k = 500  # fetch a wide net to filter down in Python
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            where=where_clause
        )
        
        matches = []
        for i in range(len(results["documents"][0])):
            doc = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            
            # Python-side temporal filtering (ISO8601 strings sort lexicographically)
            if mode == "temporal":
                msg_time = meta["timestamp"]
                if start_date and msg_time < start_date:
                    continue
                if end_date and msg_time > end_date:
                    continue
            
            # Fetch surrounding context from cached dataset
            context = []
            idx = self.msg_id_to_idx.get(meta["message_id"])
            if idx is not None:
                start_idx = max(0, idx - context_n)
                end_idx = min(len(self.messages), idx + context_n + 1)
                
                for j in range(start_idx, end_idx):
                    c_msg = self.messages[j].copy()
                    c_msg["is_match"] = (j == idx)
                    context.append(c_msg)
            
            matches.append({
                "match": {
                    "id": meta["message_id"],
                    "sender": meta["sender"],
                    "timestamp": meta["timestamp"],
                    "text": doc,
                    "score": dist
                },
                "context": context
            })
            
            # Stop once we have top_k valid matches
            if len(matches) == top_k:
                break
            
        end_time = time.time()
        time_taken_ms = round((end_time - start_time) * 1000, 2)
        
        return {
            "results": matches,
            "metrics": {
                "time_taken_ms": time_taken_ms,
                "expanded_query": expanded_query
            }
        }

# Global instance initialized lazily or explicitly
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine
