import os
import json
import time
import sys
import chromadb
from sentence_transformers import SentenceTransformer

def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print("Initializing embedding model...")
    # This will trigger the download on the first run if not cached
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    print("Model loaded successfully.")

    data_file = "data/chat_export.json"
    if not os.path.exists(data_file):
        print(f"\nFile {data_file} not found.")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        messages = json.load(f)
    
    total_msgs = len(messages)
    print(f"Loaded {total_msgs} messages from {data_file}.")
    
    if total_msgs == 0:
        return

    # Initialize ChromaDB
    print("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # We use get_or_create_collection to ensure it sets cosine space
    collection = client.get_or_create_collection(
        name="group_chat_qwen3",
        metadata={"hnsw:space": "cosine"}
    )
    
    count = collection.count()
    if count == total_msgs:
        print("Collection already contains all messages. Skipping embedding phase.")
    else:
        print("Starting full embedding and ingestion...")
        batch_size = 100
        for i in range(0, total_msgs, batch_size):
            batch = messages[i:i+batch_size]
            texts = [m["text"] for m in batch]
            ids = [m["id"] for m in batch]
            metadatas = [
                {"sender": m["sender"], "timestamp": m["timestamp"], "message_id": m["id"]} 
                for m in batch
            ]
            
            # Documents DO NOT get the instruction prefix
            embeddings = model.encode(texts, show_progress_bar=False)
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            print(f"Ingested {min(i+batch_size, total_msgs)} / {total_msgs}")
            
    # CONFIRMATION
    count = collection.count()
    print(f"\nIngestion complete. Collection count: {count} / {total_msgs}")
    
    print("\n--- PHASE 4 HARD GATE: 5 Test Queries ---")
    test_queries = [
        "Goa trip dates",
        "exam preparation and syllabus",
        "roommate pizza noise",
        "group project framework language",
        "dinner food order tonight",
        "chhuttiyon ka kya plan hai"
    ]
    
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        
        # Qwen3 requires instruction prefix for queries
        prefix = "Instruct: Given a search query, retrieve relevant chat messages that answer the query\nQuery: "
        q_embedding = model.encode(prefix + q, show_progress_bar=False)
        
        results = collection.query(
            query_embeddings=[q_embedding],
            n_results=5
        )
        
        for j in range(5):
            try:
                doc = results["documents"][0][j]
                dist = results["distances"][0][j]
                meta = results["metadatas"][0][j]
                print(f"  [{dist:.4f}] {meta['sender']}: {doc}")
            except IndexError:
                pass

if __name__ == "__main__":
    main()
